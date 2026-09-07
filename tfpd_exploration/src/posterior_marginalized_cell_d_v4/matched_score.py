"""No-data contract for the PMC-D B128 matched-score successor.

V4 is deliberately a scorer-only successor.  It wraps the completed PMC-D
producer and the immutable V3 failure/input authority; it neither changes nor
reopens the training route.  The module is stdlib-only so the public CLI can
remain dry without importing Torch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.posterior_marginalized_cell_d_v1 import matched_score as v1
from src.posterior_marginalized_cell_d_v3 import matched_score as v3


CELL = v1.CELL
PHASE = "POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_V4_B128"
SCHEMA = "posterior_marginalized_cell_d_matched_score_v4"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_V4_20260823.md"
# Replaced by the final workorder digest before this route is frozen.
WORKORDER_SHA256 = "3bd17c277e819809ac342f4e02099e4b1e08df83d4ca02223d70107a919d4b88"

FAILED_V3_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_v3"
AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_authority_v4"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_v4"

WITHIN = v1.WITHIN
EXTERNAL = v1.EXTERNAL
SURFACES = (WITHIN, EXTERNAL)
SYSTEM_PMC = v1.SYSTEM_PMC
SYSTEM_SEALED = v1.SYSTEM_SEALED
SYSTEMS = (SYSTEM_PMC, SYSTEM_SEALED)
BUDGETS = (30, 4)  # historical bridge/safety first, then headline short prefix
EVAL_BATCH_SIZE = 128
METRIC_CONTRACT = dict(v1.METRIC_CONTRACT)
INFERENCE_SEMANTICS = v1.INFERENCE_SEMANTICS
LAUNCH_ENVIRONMENT = dict(v3.LAUNCH_ENVIRONMENT)

V4_LOCAL_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/posterior_marginalized_cell_d_v4/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v4/matched_score.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v4/matched_score_physical.py",
    "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score_v4.py",
    "tfpd_exploration/tests/test_posterior_marginalized_cell_d_v4.py",
    # V4 alone invokes the historical spintshape constructor rather than the
    # producer-native pop_robust constructor.  Bind every direct repository
    # module on that new import path explicitly; no glob or ambient source
    # tree is an execution authority.
    "tfpd_exploration/src/tfpd/spintshape_module.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)


class ScoreV4Error(RuntimeError):
    """Raised for a V4 provenance, score, or lifecycle boundary violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreV4Error(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ScoreV4Error(f"{label} must be an exact lowercase SHA-256")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ScoreV4Error(f"{label} must be finite")
    return float(value)


def _read_regular_no_follow(path: Path) -> tuple[bytes, str]:
    """Read closure bytes through one regular non-symlink descriptor."""
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ScoreV4Error(f"V4 closure path inaccessible: {path}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ScoreV4Error(f"V4 closure path is not a regular non-symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ScoreV4Error(f"V4 closure path cannot be opened: {path}") from error
    identity = (int(before.st_dev), int(before.st_ino), int(before.st_size))
    try:
        opened = os.fstat(descriptor)
        if (int(opened.st_dev), int(opened.st_ino), int(opened.st_size)) != identity:
            raise ScoreV4Error(f"V4 closure inode drift: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        body = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(path)
    except OSError as error:
        raise ScoreV4Error(f"V4 closure path disappeared: {path}") from error
    if (int(after.st_dev), int(after.st_ino), int(after.st_size)) != identity or stat.S_ISLNK(after.st_mode):
        raise ScoreV4Error(f"V4 closure path changed while read: {path}")
    return body, _digest(body)


@dataclass(frozen=True)
class V4ScoreCell:
    surface: str
    budget: int
    system: str

    def payload(self) -> dict[str, object]:
        _require(self.surface in SURFACES, "V4 score cell surface drift")
        _require(self.budget in BUDGETS, "V4 score cell budget drift")
        _require(self.system in SYSTEMS, "V4 score cell system drift")
        return {
            "surface": self.surface,
            "budget": self.budget,
            "system": self.system,
            "carrier": INFERENCE_SEMANTICS,
            "eval_batch_size": EVAL_BATCH_SIZE,
        }


def score_matrix() -> tuple[V4ScoreCell, ...]:
    return tuple(
        V4ScoreCell(surface, budget, system)
        for surface in SURFACES
        for budget in BUDGETS
        for system in SYSTEMS
    )


@dataclass(frozen=True)
class V4ImplementationClosure:
    """Explicit V3 base closure plus additive V4 leaves and V3 failure graph."""

    v3_base: Mapping[str, object]
    local_sha256_by_path: Mapping[str, str]
    failed_v3_graph: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        if set(self.local_sha256_by_path) != set(V4_LOCAL_CLOSURE):
            raise ScoreV4Error("V4 local closure topology drift")
        hashes = {path: _sha(self.local_sha256_by_path[path], f"V4 closure {path}") for path in V4_LOCAL_CLOSURE}
        if hashes[WORKORDER_RELATIVE] != WORKORDER_SHA256:
            raise ScoreV4Error("V4 workorder SHA drift")
        if not isinstance(self.v3_base, Mapping) or not isinstance(self.failed_v3_graph, Mapping):
            raise ScoreV4Error("V4 base closure/predecessor schema drift")
        body = {
            "schema": "posterior_marginalized_cell_d_matched_score_v4_closure_v1",
            "v3_base_closure": dict(self.v3_base),
            "local_paths": list(V4_LOCAL_CLOSURE),
            "local_sha256_by_path": hashes,
            "failed_v3_root_relative": FAILED_V3_ROOT_RELATIVE,
            "failed_v3_graph": dict(self.failed_v3_graph),
            "evaluation_batch_size": EVAL_BATCH_SIZE,
            "score_matrix": [cell.payload() for cell in score_matrix()],
            "launch_environment": dict(LAUNCH_ENVIRONMENT),
        }
        return {**body, "closure_sha256": _digest(_json(body))}


def implementation_closure(
    root: Path,
    *,
    v3_base_closure: Mapping[str, object],
    failed_v3_graph: Mapping[str, object],
) -> V4ImplementationClosure:
    root = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in V4_LOCAL_CLOSURE:
        _body, digest = _read_regular_no_follow(root / relative)
        hashes[relative] = digest
    return V4ImplementationClosure(
        v3_base=dict(v3_base_closure), local_sha256_by_path=hashes,
        failed_v3_graph=dict(failed_v3_graph),
    )


@dataclass(frozen=True)
class V4ScoreIdentity:
    """V4 identity wraps, rather than mutates, V1 producer/sealed provenance."""

    base_identity: v1.ScoreIdentity
    closure: V4ImplementationClosure
    v3_input_authority_sha256: str

    def payload(self) -> dict[str, object]:
        base = self.base_identity.payload()
        closure = self.closure.payload()
        return {
            "schema": "posterior_marginalized_cell_d_matched_score_v4_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "base_identity": base,
            "closure": closure,
            "v3_input_authority_sha256": _sha(self.v3_input_authority_sha256, "V3 input authority SHA"),
            "score_matrix": [cell.payload() for cell in score_matrix()],
            "metric": dict(METRIC_CONTRACT),
            "inference": INFERENCE_SEMANTICS,
        }


@dataclass(frozen=True)
class V4SessionScore:
    session: str
    n_windows: int
    r2: float
    prediction_sha256: str
    input_record_sha256: str

    def payload(self) -> dict[str, object]:
        _require(isinstance(self.session, str) and self.session and "/" not in self.session, "V4 score session drift")
        _require(type(self.n_windows) is int and self.n_windows > 0, "V4 n_windows drift")
        return {
            "session": self.session,
            "n_windows": self.n_windows,
            "r2": _finite(self.r2, "V4 session R2"),
            "prediction_sha256": _sha(self.prediction_sha256, "V4 prediction SHA"),
            "input_record_sha256": _sha(self.input_record_sha256, "V4 input-record SHA"),
        }


@dataclass(frozen=True)
class V4CellEvidence:
    cell: V4ScoreCell
    sessions: tuple[V4SessionScore, ...]
    model_swa_sha256: str
    state_before_sha256: str
    state_after_sha256: str

    def payload(self, *, identity: V4ScoreIdentity, input_sha256: str) -> dict[str, object]:
        cell = self.cell.payload()
        expected = identity.base_identity.within_roster if self.cell.surface == WITHIN else identity.base_identity.external_roster
        rows = [row.payload() for row in self.sessions]
        if [row["session"] for row in rows] != list(expected):
            raise ScoreV4Error("V4 score evidence session/order drift")
        return {
            "cell": cell,
            "sessions": rows,
            "model_swa_sha256": _sha(self.model_swa_sha256, "V4 model SWA SHA"),
            "model_state_before_sha256": _sha(self.state_before_sha256, "V4 state-before SHA"),
            "model_state_after_sha256": _sha(self.state_after_sha256, "V4 state-after SHA"),
            "state_unchanged": self.state_before_sha256 == self.state_after_sha256,
            "eval_mode": True,
            "dropout_disabled": True,
            "gradients_none": True,
            "same_materialized_input": True,
            "input_authority_sha256": _sha(input_sha256, "V4 input authority SHA"),
        }


def _paired_summary(pmc: Sequence[Mapping[str, object]], sealed: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if [row.get("session") for row in pmc] != [row.get("session") for row in sealed]:
        raise ScoreV4Error("V4 paired-session order drift")
    deltas = [_finite(left.get("r2"), "PMC R2") - _finite(right.get("r2"), "sealed R2")
              for left, right in zip(pmc, sealed, strict=True)]
    ordered = sorted(deltas)
    n = len(deltas)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {
        "mean": float(sum(deltas) / n),
        "median": float(median),
        "n_positive": int(sum(value > 0.0 for value in deltas)),
        "n_sessions": n,
        "deltas": deltas,
    }


def build_score_payload(
    *, identity: V4ScoreIdentity, input_payload: Mapping[str, object], evidence: Sequence[V4CellEvidence],
    historical_bridge: Mapping[str, object],
) -> dict[str, object]:
    identity_payload = identity.payload()
    input_sha = _digest(_json(dict(input_payload)))
    cells = list(evidence)
    if [item.cell for item in cells] != list(score_matrix()):
        raise ScoreV4Error("V4 evidence matrix/order drift")
    cell_payloads = [item.payload(identity=identity, input_sha256=input_sha) for item in cells]
    by_key = {(item["cell"]["surface"], item["cell"]["budget"], item["cell"]["system"]): item for item in cell_payloads}
    paired: dict[str, object] = {}
    summaries: dict[str, object] = {}
    for surface in SURFACES:
        for budget in BUDGETS:
            pmc = by_key[(surface, budget, SYSTEM_PMC)]["sessions"]
            sealed = by_key[(surface, budget, SYSTEM_SEALED)]["sessions"]
            if not isinstance(pmc, list) or not isinstance(sealed, list):
                raise ScoreV4Error("V4 cell session schema drift")
            paired[f"{surface}_M{budget}_pmc_minus_sealed"] = _paired_summary(pmc, sealed)
            for system, rows in ((SYSTEM_PMC, pmc), (SYSTEM_SEALED, sealed)):
                values = [_finite(row.get("r2"), "V4 session R2") for row in rows]
                ordered = sorted(values)
                n = len(values)
                summaries[f"{surface}_M{budget}_{system}"] = {
                    "mean_r2": float(sum(values) / n),
                    "median_r2": float(ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0),
                    "n_sessions": n,
                }
    payload = {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_score_v1",
        "identity": identity_payload,
        "input_authority_sha256": input_sha,
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "metric": dict(METRIC_CONTRACT),
        "cells": cell_payloads,
        "equal_session_summaries": summaries,
        "paired_pmc_minus_sealed": paired,
        "historical_sealed_m30_bridge": dict(historical_bridge),
        "historical_absolute_reference": (
            "exact_reproduced" if historical_bridge.get("all_exact") is True
            else "contextual_non_authorizing__same_evaluator_paired_screen_governs"
        ),
        "boundaries": {
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "formal_opened": False,
            "posterior_inference_used": False,
            "normalizer_refit": False,
        },
    }
    validate_score_payload(payload, identity=identity)
    return payload


def validate_score_payload(payload: Mapping[str, object], *, identity: V4ScoreIdentity) -> None:
    required = {
        "schema", "identity", "input_authority_sha256", "evaluation_batch_size", "metric", "cells",
        "equal_session_summaries", "paired_pmc_minus_sealed", "historical_sealed_m30_bridge",
        "historical_absolute_reference", "boundaries",
    }
    if set(payload) != required or payload.get("schema") != "posterior_marginalized_cell_d_matched_score_v4_score_v1":
        raise ScoreV4Error("V4 score payload schema drift")
    if payload.get("identity") != identity.payload() or payload.get("evaluation_batch_size") != EVAL_BATCH_SIZE:
        raise ScoreV4Error("V4 score identity/batch drift")
    _sha(payload.get("input_authority_sha256"), "V4 score input SHA")
    if payload.get("metric") != METRIC_CONTRACT:
        raise ScoreV4Error("V4 score metric drift")
    input_sha = _sha(payload.get("input_authority_sha256"), "V4 score input SHA")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != len(score_matrix()):
        raise ScoreV4Error("V4 score matrix count drift")
    if [row.get("cell") for row in cells if isinstance(row, Mapping)] != [cell.payload() for cell in score_matrix()]:
        raise ScoreV4Error("V4 score matrix topology drift")
    cell_by_key: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for cell, raw in zip(score_matrix(), cells, strict=True):
        if not isinstance(raw, Mapping):
            raise ScoreV4Error("V4 score cell payload schema drift")
        required_cell = {
            "cell", "sessions", "model_swa_sha256", "model_state_before_sha256", "model_state_after_sha256",
            "state_unchanged", "eval_mode", "dropout_disabled", "gradients_none", "same_materialized_input",
            "input_authority_sha256",
        }
        if set(raw) != required_cell or raw.get("cell") != cell.payload():
            raise ScoreV4Error("V4 score cell schema/cell drift")
        if raw.get("input_authority_sha256") != input_sha:
            raise ScoreV4Error("V4 score cell input-authority binding drift")
        for label in ("model_swa_sha256", "model_state_before_sha256", "model_state_after_sha256"):
            _sha(raw.get(label), f"V4 score cell {label}")
        if raw.get("model_state_before_sha256") != raw.get("model_state_after_sha256") \
                or any(raw.get(flag) is not True for flag in (
                    "state_unchanged", "eval_mode", "dropout_disabled", "gradients_none", "same_materialized_input",
                )):
            raise ScoreV4Error("V4 score cell state/eval boundary drift")
        expected = identity.base_identity.within_roster if cell.surface == WITHIN else identity.base_identity.external_roster
        sessions = raw.get("sessions")
        if not isinstance(sessions, list) or len(sessions) != len(expected):
            raise ScoreV4Error("V4 score cell session count drift")
        expected_rows: list[dict[str, object]] = []
        for name, row in zip(expected, sessions, strict=True):
            if not isinstance(row, Mapping) or set(row) != {
                "session", "n_windows", "r2", "prediction_sha256", "input_record_sha256",
            } or row.get("session") != name or type(row.get("n_windows")) is not int or row["n_windows"] <= 0:
                raise ScoreV4Error("V4 score session schema/order drift")
            _finite(row.get("r2"), "V4 score session R2")
            _sha(row.get("prediction_sha256"), "V4 score prediction SHA")
            _sha(row.get("input_record_sha256"), "V4 score input-record SHA")
            expected_rows.append(dict(row))
        cell_by_key[(cell.surface, cell.budget, cell.system)] = raw
    bridge = payload.get("historical_sealed_m30_bridge")
    required_bridge = {
        "schema", "evaluation_batch_size", "metric", "sealed_swa_sha256", "sealed_baseline_receipt_sha256",
        "input_authority_sha256", "rows", "all_exact", "first_mismatch", "interpretation",
    }
    if not isinstance(bridge, Mapping) or set(bridge) != required_bridge \
            or bridge.get("schema") != "posterior_marginalized_cell_d_historical_sealed_bridge_v4" \
            or bridge.get("evaluation_batch_size") != EVAL_BATCH_SIZE \
            or bridge.get("metric") != METRIC_CONTRACT or bridge.get("input_authority_sha256") != input_sha:
        raise ScoreV4Error("V4 historical bridge schema/batch drift")
    _sha(bridge.get("sealed_swa_sha256"), "V4 bridge sealed SWA SHA")
    _sha(bridge.get("sealed_baseline_receipt_sha256"), "V4 bridge sealed baseline SHA")
    bridge_rows = bridge.get("rows")
    if not isinstance(bridge_rows, Mapping) or set(bridge_rows) != set(SURFACES):
        raise ScoreV4Error("V4 historical bridge surface topology drift")
    first = bridge.get("first_mismatch")
    all_exact = bridge.get("all_exact")
    if type(all_exact) is not bool or (all_exact and first is not None) or (not all_exact and not isinstance(first, Mapping)):
        raise ScoreV4Error("V4 historical bridge mismatch semantics drift")
    observed_first: Mapping[str, object] | None = None
    for surface in SURFACES:
        rows = bridge_rows[surface]
        expected = identity.base_identity.within_roster if surface == WITHIN else identity.base_identity.external_roster
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ScoreV4Error("V4 historical bridge row count drift")
        for name, row in zip(expected, rows, strict=True):
            if not isinstance(row, Mapping) or set(row) != {
                "session", "historical_n_windows", "live_n_windows", "historical_r2", "live_r2",
                "prediction_sha256", "exact_match", "mismatch_field",
            } or row.get("session") != name or type(row.get("historical_n_windows")) is not int \
                    or type(row.get("live_n_windows")) is not int \
                    or row["historical_n_windows"] <= 0 or row["live_n_windows"] <= 0:
                raise ScoreV4Error("V4 historical bridge row schema/order drift")
            _finite(row.get("historical_r2"), "V4 historical R2")
            _finite(row.get("live_r2"), "V4 live bridge R2")
            _sha(row.get("prediction_sha256"), "V4 bridge prediction SHA")
            exact = row.get("historical_n_windows") == row.get("live_n_windows") \
                and row.get("historical_r2") == row.get("live_r2")
            if row.get("exact_match") is not exact or (exact and row.get("mismatch_field") is not None) \
                    or (not exact and row.get("mismatch_field") not in {"n_windows", "r2"}):
                raise ScoreV4Error("V4 historical bridge exactness row drift")
            if not exact and observed_first is None:
                observed_first = {"surface": surface, **dict(row)}
    if (observed_first is None) != all_exact or (observed_first is not None and dict(first) != dict(observed_first)):
        raise ScoreV4Error("V4 historical bridge first-mismatch binding drift")
    if payload.get("historical_absolute_reference") not in {
        "exact_reproduced", "contextual_non_authorizing__same_evaluator_paired_screen_governs",
    }:
        raise ScoreV4Error("V4 historical-reference label drift")
    expected_reference = "exact_reproduced" if all_exact else "contextual_non_authorizing__same_evaluator_paired_screen_governs"
    if payload.get("historical_absolute_reference") != expected_reference:
        raise ScoreV4Error("V4 historical-reference label/bridge drift")
    summaries = payload.get("equal_session_summaries")
    paired = payload.get("paired_pmc_minus_sealed")
    if not isinstance(summaries, Mapping) or not isinstance(paired, Mapping):
        raise ScoreV4Error("V4 paired summary schema drift")
    expected_summary_keys = {f"{surface}_M{budget}_{system}" for surface in SURFACES for budget in BUDGETS for system in SYSTEMS}
    expected_paired_keys = {f"{surface}_M{budget}_pmc_minus_sealed" for surface in SURFACES for budget in BUDGETS}
    if set(summaries) != expected_summary_keys or set(paired) != expected_paired_keys:
        raise ScoreV4Error("V4 paired summary topology drift")
    for surface in SURFACES:
        for budget in BUDGETS:
            pmc_rows = cell_by_key[(surface, budget, SYSTEM_PMC)]["sessions"]
            sealed_rows = cell_by_key[(surface, budget, SYSTEM_SEALED)]["sessions"]
            if not isinstance(pmc_rows, list) or not isinstance(sealed_rows, list):
                raise ScoreV4Error("V4 paired source rows drift")
            exact_pair = _paired_summary(pmc_rows, sealed_rows)
            if paired[f"{surface}_M{budget}_pmc_minus_sealed"] != exact_pair:
                raise ScoreV4Error("V4 paired delta summary drift")
            for system, rows in ((SYSTEM_PMC, pmc_rows), (SYSTEM_SEALED, sealed_rows)):
                values = [_finite(row.get("r2"), "V4 summary R2") for row in rows]
                ordered = sorted(values)
                n = len(values)
                expected_summary = {
                    "mean_r2": float(sum(values) / n),
                    "median_r2": float(ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0),
                    "n_sessions": n,
                }
                if summaries[f"{surface}_M{budget}_{system}"] != expected_summary:
                    raise ScoreV4Error("V4 equal-session summary drift")
    boundaries = payload.get("boundaries")
    expected_boundaries = {
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "formal_opened": False, "posterior_inference_used": False, "normalizer_refit": False,
    }
    if boundaries != expected_boundaries:
        raise ScoreV4Error("V4 score boundary drift")


def dry_plan() -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_plan_v1",
        "status": "DRY_ONLY__NO_NWB_NO_CUDA_NO_ROOT_RESERVATION__ROOT_REVIEW_REQUIRED",
        "failed_v3_root": FAILED_V3_ROOT_RELATIVE,
        "authority_root": AUTHORITY_ROOT_RELATIVE,
        "score_root": SCORE_ROOT_RELATIVE,
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "score_matrix": [cell.payload() for cell in score_matrix()],
        "historical_bridge": "B128 exact comparison; mismatch is persisted contextual evidence, never tolerance-widened",
        "launch_environment": dict(LAUNCH_ENVIRONMENT),
    }


def assert_fresh_prospective_roots(root: Path) -> None:
    """Fail closed if either future V4 namespace already exists or aliases a symlink.

    This is intentionally read-only.  It is used by root review immediately
    before reservation; the public dry CLI never calls it.
    """
    base = Path(root).absolute()
    for relative in (AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE):
        path = base / relative
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ScoreV4Error(f"V4 prospective root cannot be inspected: {relative}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ScoreV4Error(f"V4 prospective root is an unsafe symlink: {relative}")
        raise ScoreV4Error(f"V4 prospective root is not fresh: {relative}")
