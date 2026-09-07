"""Descriptor-safe authority and shared-lifecycle hooks for the O1/O2 smoke.

The route owns only its new provenance/receipt codec.  Physical parser,
model, causal transition, metric, and immutable publication order remain in
the reviewed Precision-V2/V8/V5 and common score-lifecycle implementations.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.causal_dual_memory_cell_d_score_v1 import score as shared_score
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import plan as precision_v1_plan

from . import plan


AUTHORITY_TOPOLOGY = shared_score.AUTHORITY_TOPOLOGY
SCORE_TOPOLOGY = shared_score.SCORE_TOPOLOGY


class SpeedSmokeError(RuntimeError):
    """Fail closed for speed-smoke lineage, receipt, or lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpeedSmokeError(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha256(value, label)
    except plan.SpeedSmokePlanError as error:
        raise SpeedSmokeError(str(error)) from error


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SpeedSmokeError(f"speed-smoke {label} directory is inaccessible") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SpeedSmokeError(f"speed-smoke {label} directory is noncanonical")
    return int(info.st_dev), int(info.st_ino)


def _no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(value, int) or value == 0:
        raise SpeedSmokeError("speed-smoke immutable reader requires O_NOFOLLOW")
    return value


def _read_held_leaf(fd: int, name: str, *, label: str) -> bytes:
    """Read a regular 0444/nlink1 basename through its already-held root FD."""
    if not isinstance(name, str) or Path(name).name != name:
        raise SpeedSmokeError("speed-smoke held leaf name drift")
    descriptor = -1
    try:
        descriptor = os.open(
            name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _no_follow_flag(), dir_fd=fd,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o444 or int(info.st_nlink) != 1
        ):
            raise SpeedSmokeError(f"speed-smoke {label} leaf mode/link drift: {name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    except SpeedSmokeError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise SpeedSmokeError(f"speed-smoke {label} leaf is a symlink: {name}") from error
        raise SpeedSmokeError(f"speed-smoke {label} held leaf is inaccessible: {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_exact_pair(fd: int, name: str, expected_sha256: str, *, label: str) -> dict[str, object]:
    _sha(expected_sha256, f"{label} {name}")
    body = _read_held_leaf(fd, name, label=label)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = _read_held_leaf(fd, f"{name}.sha256", label=label)
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise SpeedSmokeError(f"speed-smoke {label} canonical sidecar drift: {name}")
    if digest != expected_sha256:
        raise SpeedSmokeError(f"speed-smoke {label} body SHA drift: {name}")
    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise SpeedSmokeError(f"speed-smoke {label} body is malformed JSON: {name}") from error
    if not isinstance(payload, Mapping):
        raise SpeedSmokeError(f"speed-smoke {label} JSON root drift: {name}")
    return dict(payload)


def _read_exact_directory(
    directory: Path, *, label: str, expected_sha256s: Mapping[str, str],
) -> tuple[tuple[int, int], dict[str, dict[str, object]]]:
    """Hold one exact immutable directory and reject extra/missing leaves."""
    named = _directory_identity(directory, label)
    descriptor = -1
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _no_follow_flag())
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise SpeedSmokeError(f"speed-smoke {label} directory identity drift before read")
        expected_names = set(expected_sha256s) | {f"{name}.sha256" for name in expected_sha256s}
        if set(os.listdir(descriptor)) != expected_names:
            raise SpeedSmokeError(f"speed-smoke {label} exact pair topology drift")
        bodies = {
            name: _read_exact_pair(descriptor, name, expected_sha256s[name], label=label)
            for name in sorted(expected_sha256s)
        }
        if _directory_identity(directory, label) != named:
            raise SpeedSmokeError(f"speed-smoke {label} directory identity drift during read")
        return named, bodies
    except SpeedSmokeError:
        raise
    except OSError as error:
        raise SpeedSmokeError(f"speed-smoke {label} descriptor read failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _row_from_score(score_body: Mapping[str, object], witness: plan.SmokeRowWitness) -> dict[str, object]:
    """Reconstruct exactly one historical V2 row rather than spot-check it."""
    summaries = score_body.get("budget_summaries")
    if not isinstance(summaries, Mapping):
        raise SpeedSmokeError("completed V2 score budget summaries are absent")
    summary = summaries.get(str(witness.budget))
    if not isinstance(summary, Mapping) or summary.get("budget") != witness.budget:
        raise SpeedSmokeError("completed V2 selected budget summary drift")
    cells = summary.get("precision_cells")
    if not isinstance(cells, list):
        raise SpeedSmokeError("completed V2 precision cells are absent")
    selected_cells = [
        cell for cell in cells
        if isinstance(cell, Mapping)
        and cell.get("surface") == witness.surface
        and cell.get("budget") == witness.budget
    ]
    if len(selected_cells) != 1:
        raise SpeedSmokeError("completed V2 selected cell topology drift")
    sessions = selected_cells[0].get("sessions")
    if not isinstance(sessions, list):
        raise SpeedSmokeError("completed V2 selected cell sessions are absent")
    rows = [row for row in sessions if isinstance(row, Mapping) and row.get("session") == witness.session]
    if len(rows) != 1:
        raise SpeedSmokeError("completed V2 selected session row topology drift")
    row = dict(rows[0])
    if _digest(row) != witness.canonical_row_sha256:
        raise SpeedSmokeError("completed V2 selected row canonical digest drift")
    if (
        row.get("budget") != witness.budget
        or row.get("governing_r2") != witness.governing_r2
        or row.get("prediction_sha256") != witness.prediction_sha256
        or row.get("input_record_sha256") != witness.input_record_sha256
        or row.get("n_windows") != witness.n_windows
        or row.get("model_state_before_sha256") != witness.model_state_sha256
        or row.get("model_state_after_sha256") != witness.model_state_sha256
        or row.get("sealed_model_load_proof_sha256") != witness.sealed_model_load_proof_sha256
    ):
        raise SpeedSmokeError("completed V2 selected row literal witness drift")
    transitions = row.get("transition_records")
    if not isinstance(transitions, list) or len(transitions) != witness.transition_count:
        raise SpeedSmokeError("completed V2 selected row transition topology drift")
    commits = sum(item.get("carrier_transition_committed") is True for item in transitions if isinstance(item, Mapping))
    if commits != witness.carrier_commits:
        raise SpeedSmokeError("completed V2 selected row carrier-commit count drift")
    return row


def _validate_historical_v2_semantics(
    authority: Mapping[str, Mapping[str, object]], score_graph: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object], tuple[dict[str, object], ...]]:
    """Check stable graph links without rebuilding V2's historical closure today."""
    preflight = authority["official_preflight.json"]
    authorization = authority["root_authorization.json"]
    attempt = score_graph["attempt.json"]
    input_authority = score_graph["input_authority.json"]
    score_body = score_graph["score.json"]
    terminal = score_graph["terminal.json"]
    identity = attempt.get("identity")
    if not isinstance(identity, Mapping) or _digest(identity) != plan.V2_ATTEMPT_IDENTITY_SHA256:
        raise SpeedSmokeError("completed V2 attempt identity digest drift")
    closure = identity.get("closure")
    if not isinstance(closure, Mapping) or closure.get("closure_sha256") != plan.V2_FINAL_CLOSURE_SHA256:
        raise SpeedSmokeError("completed V2 historical closure drift")
    if identity.get("selected_device_profile") != plan.HISTORICAL_V2_SELECTED_PROFILE:
        raise SpeedSmokeError("completed V2 historical GPU1 device evidence drift")
    if (
        preflight.get("identity") != dict(identity)
        or authorization.get("official_preflight_sha256") != plan.V2_AUTHORITY_SHAS["official_preflight.json"]
        or authorization.get("identity_sha256") != plan.V2_ATTEMPT_IDENTITY_SHA256
        or attempt.get("preflight_sha256") != plan.V2_AUTHORITY_SHAS["official_preflight.json"]
        or attempt.get("authorization_sha256") != plan.V2_AUTHORITY_SHAS["root_authorization.json"]
        or score_body.get("identity") != dict(identity)
        or score_body.get("input_authority_sha256") != plan.V2_RESULT_SHAS["input_authority.json"]
        or input_authority.get("identity_sha256") != plan.V2_ATTEMPT_IDENTITY_SHA256
        or terminal.get("identity") != dict(identity)
        or terminal.get("attempt_sha256") != plan.V2_RESULT_SHAS["attempt.json"]
        or terminal.get("input_authority_sha256") != plan.V2_RESULT_SHAS["input_authority.json"]
        or terminal.get("score_sha256") != plan.V2_RESULT_SHAS["score.json"]
        or terminal.get("launch_closure_sha256") != plan.V2_FINAL_CLOSURE_SHA256
        or terminal.get("final_closure_sha256") != plan.V2_FINAL_CLOSURE_SHA256
        or terminal.get("target_optimizer_backward_update") != 0
    ):
        raise SpeedSmokeError("completed V2 authority/result graph linkage drift")
    # Input records are full authority evidence; prove the two selected rows
    # are the same canonical record used by each historical cell.
    records = input_authority.get("records")
    if not isinstance(records, list):
        raise SpeedSmokeError("completed V2 input authority records are absent")
    reconstructed = tuple(_row_from_score(score_body, witness) for witness in plan.SMOKE_ROWS)
    for witness in plan.SMOKE_ROWS:
        matches = [
            record for record in records
            if isinstance(record, Mapping) and record.get("surface") == witness.surface
            and record.get("session") == witness.session
        ]
        if len(matches) != 1 or _digest(dict(matches[0])) != witness.input_record_sha256:
            raise SpeedSmokeError("completed V2 selected input record reconstruction drift")
    return dict(identity), dict(input_authority), reconstructed


@dataclass(frozen=True)
class CompletedV2Binding:
    """Held descriptor identities plus semantic V2 graph and row witnesses."""

    authority_directory_identity: tuple[int, int]
    score_directory_identity: tuple[int, int]
    historical_v2_identity: Mapping[str, object]
    historical_v2_input_authority: Mapping[str, object]
    historical_rows: tuple[Mapping[str, object], ...]

    def payload(self) -> dict[str, object]:
        _require(
            all(type(value) is int and value > 0 for value in (*self.authority_directory_identity, *self.score_directory_identity)),
            "completed V2 directory identity drift",
        )
        identity = dict(self.historical_v2_identity)
        _require(_digest(identity) == plan.V2_ATTEMPT_IDENTITY_SHA256,
                 "completed V2 binding identity digest drift")
        rows = [dict(value) for value in self.historical_rows]
        input_authority = dict(self.historical_v2_input_authority)
        if not isinstance(input_authority.get("records"), list):
            raise SpeedSmokeError("completed V2 binding input-authority records are absent")
        if len(rows) != len(plan.SMOKE_ROWS):
            raise SpeedSmokeError("completed V2 binding selected-row cardinality drift")
        for raw, witness in zip(rows, plan.SMOKE_ROWS, strict=True):
            if _digest(raw) != witness.canonical_row_sha256:
                raise SpeedSmokeError("completed V2 binding selected-row digest drift")
        body = {
            "schema": "precision_aware_cdmd_speed_smoke_completed_v2_binding_v1",
            "contract": plan.COMPLETED_V2.payload(),
            "authority_directory_identity": list(self.authority_directory_identity),
            "score_directory_identity": list(self.score_directory_identity),
            "historical_v2_identity": identity,
            "historical_v2_identity_sha256": plan.V2_ATTEMPT_IDENTITY_SHA256,
            "historical_v2_input_authority": input_authority,
            "historical_v2_input_authority_sha256": plan.V2_RESULT_SHAS["input_authority.json"],
            "historical_selected_rows": rows,
            "selected_row_witnesses": [item.payload() for item in plan.SMOKE_ROWS],
        }
        return {**body, "binding_sha256": _digest(body)}


def validate_completed_v2_binding(value: object) -> dict[str, object]:
    required = {
        "schema", "contract", "authority_directory_identity", "score_directory_identity",
        "historical_v2_identity", "historical_v2_identity_sha256", "historical_v2_input_authority",
        "historical_v2_input_authority_sha256", "historical_selected_rows",
        "selected_row_witnesses", "binding_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "precision_aware_cdmd_speed_smoke_completed_v2_binding_v1":
        raise SpeedSmokeError("completed V2 binding schema drift")
    if value.get("contract") != plan.COMPLETED_V2.payload() or value.get("selected_row_witnesses") != [item.payload() for item in plan.SMOKE_ROWS]:
        raise SpeedSmokeError("completed V2 binding contract/witness drift")
    directories: list[tuple[int, int]] = []
    for label in ("authority_directory_identity", "score_directory_identity"):
        raw = value.get(label)
        if not isinstance(raw, list) or len(raw) != 2 or any(type(item) is not int or item <= 0 for item in raw):
            raise SpeedSmokeError(f"completed V2 binding {label} drift")
        directories.append((int(raw[0]), int(raw[1])))
    identity = value.get("historical_v2_identity")
    input_authority = value.get("historical_v2_input_authority")
    rows = value.get("historical_selected_rows")
    if (
        not isinstance(identity, Mapping) or value.get("historical_v2_identity_sha256") != plan.V2_ATTEMPT_IDENTITY_SHA256
        or _digest(dict(identity)) != plan.V2_ATTEMPT_IDENTITY_SHA256
        or not isinstance(input_authority, Mapping) or not isinstance(input_authority.get("records"), list)
        or value.get("historical_v2_input_authority_sha256") != plan.V2_RESULT_SHAS["input_authority.json"]
        or not isinstance(rows, list)
    ):
        raise SpeedSmokeError("completed V2 binding historical identity/rows drift")
    result = CompletedV2Binding(
        authority_directory_identity=directories[0], score_directory_identity=directories[1],
        historical_v2_identity=dict(identity), historical_v2_input_authority=dict(input_authority),
        historical_rows=tuple(dict(item) for item in rows if isinstance(item, Mapping)),
    ).payload()
    if len(result["historical_selected_rows"]) != len(rows) or dict(value) != result:
        raise SpeedSmokeError("completed V2 binding canonical drift")
    return result


def validate_completed_precision_v2(root: Path) -> CompletedV2Binding:
    """Descriptor-read the accepted V2 12-leaf graph before any new action."""
    base = Path(root).absolute()
    authority_identity, authority = _read_exact_directory(
        base / plan.V2_AUTHORITY_ROOT_RELATIVE, label="completed V2 authority",
        expected_sha256s=plan.V2_AUTHORITY_SHAS,
    )
    score_identity, score_graph = _read_exact_directory(
        base / plan.V2_SCORE_ROOT_RELATIVE, label="completed V2 score",
        expected_sha256s=plan.V2_RESULT_SHAS,
    )
    historical_identity, input_authority, rows = _validate_historical_v2_semantics(authority, score_graph)
    return CompletedV2Binding(
        authority_directory_identity=authority_identity, score_directory_identity=score_identity,
        historical_v2_identity=historical_identity, historical_v2_input_authority=input_authority, historical_rows=rows,
    )


def _runtime_v1_identity_from_completed(binding: Mapping[str, object]) -> precision_v1_plan.ScoreIdentity:
    checked = validate_completed_v2_binding(binding)
    historical = checked["historical_v2_identity"]
    if not isinstance(historical, Mapping):  # defensive after typed validator
        raise SpeedSmokeError("completed V2 historical identity is absent")
    science = historical.get("historical_v1_science_identity")
    if not isinstance(science, Mapping):
        raise SpeedSmokeError("completed V2 V1 science identity is absent")
    try:
        runtime = precision_v1_plan.ScoreIdentity(
            closure=science["closure"], v8_binding=science["v8_predecessor_binding"],
            selected_device_profile=plan.GPU0_PROFILE,
        )
    except (KeyError, TypeError, precision_v1_plan.PrecisionMatchedScorePlanError) as error:
        raise SpeedSmokeError("completed V2 runtime V1 identity reconstruction failed") from error
    payload = runtime.payload()
    expected_science = dict(science)
    expected_science["selected_device_profile"] = dict(plan.GPU0_PROFILE)
    if payload != expected_science:
        raise SpeedSmokeError("speed-smoke GPU0 runtime science identity drift")
    return runtime


@dataclass(frozen=True)
class SpeedSmokeIdentity:
    """Current successor closure plus historical V2 and runtime bridge evidence."""

    closure: Mapping[str, object]
    completed_v2_binding: Mapping[str, object]
    runtime_v1_identity: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        closure = plan.validate_implementation_closure(self.closure)
        completed = validate_completed_v2_binding(self.completed_v2_binding)
        runtime = _runtime_v1_identity_from_completed(completed).payload()
        if dict(self.runtime_v1_identity) != runtime:
            raise SpeedSmokeError("speed-smoke runtime identity bridge drift")
        return {
            "schema": "precision_aware_cdmd_speed_smoke_identity_v1",
            "cell": plan.CELL,
            "phase": plan.PHASE,
            "closure": closure,
            "completed_precision_v2_binding": completed,
            "completed_precision_v2_binding_sha256": completed["binding_sha256"],
            "historical_v2_identity_sha256": plan.V2_ATTEMPT_IDENTITY_SHA256,
            "historical_v2_gpu1_evidence": dict(plan.HISTORICAL_V2_SELECTED_PROFILE),
            "selected_device_profile": dict(plan.GPU0_PROFILE),
            "runtime_v1_science_identity": runtime,
            "canonical_source_environment": plan.canonical_source_environment_payload(),
            "smoke_execution_order": [list(item) for item in plan.SMOKE_EXECUTION_ORDER],
            "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": plan.SCORE_ROOT_RELATIVE,
            "m30_rerun": False,
            "sealed_comparator_rerun": False,
            "non_governing_engineering_smoke": True,
            "target_updates_forbidden": True,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())

    def runtime_identity(self) -> precision_v1_plan.ScoreIdentity:
        return _runtime_v1_identity_from_completed(validate_completed_v2_binding(self.completed_v2_binding))


def build_reviewed_identity(root: Path) -> SpeedSmokeIdentity:
    """Build only from held V2 bytes and this successor's current closure."""
    completed = validate_completed_precision_v2(Path(root))
    binding = completed.payload()
    runtime = _runtime_v1_identity_from_completed(binding)
    return SpeedSmokeIdentity(
        closure=plan.implementation_closure(Path(root)).payload(),
        completed_v2_binding=binding, runtime_v1_identity=runtime.payload(),
    )


def _identity_payload(identity: SpeedSmokeIdentity) -> dict[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke requires its exact typed identity")
    return identity.payload()


def _require_live_completed_v2(root: Path, identity: SpeedSmokeIdentity) -> CompletedV2Binding:
    observed = validate_completed_precision_v2(Path(root))
    if observed.payload() != _identity_payload(identity)["completed_precision_v2_binding"]:
        raise SpeedSmokeError("speed-smoke held completed V2 predecessor/identity drift")
    return observed


def build_target_free_preflight(
    *, root: Path, identity: SpeedSmokeIdentity, predecessor: CompletedV2Binding,
    fixed_authority: shared_score.FixedEvaluationAuthority | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    environment = plan.validate_selected_launch_environment(environ)
    if predecessor.payload() != _identity_payload(identity)["completed_precision_v2_binding"]:
        raise SpeedSmokeError("speed-smoke preflight completed V2/identity drift")
    observed = _require_live_completed_v2(Path(root), identity)
    if observed.payload() != predecessor.payload():
        raise SpeedSmokeError("speed-smoke preflight held completed V2 drift")
    fixed = shared_score.derive_fixed_evaluation_authority(Path(root)) if fixed_authority is None else fixed_authority
    return {
        "schema": "precision_aware_cdmd_speed_smoke_target_free_preflight_v1",
        "status": "PREFLIGHT_ACCEPTED",
        "identity": _identity_payload(identity),
        # The shared immutable lifecycle names its non-target predecessor
        # field ``source_gate``.  Here it is exactly the completed V2 binding,
        # not a relabelled source-training receipt or caller mapping.
        "source_gate": predecessor.payload(),
        "completed_precision_v2": predecessor.payload(),
        "evaluation_authority": fixed.payload(),
        "metric": dict(base_plan_metric()),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "canonical_launch_environment": environment,
        "target_free": True,
        "target_paths_resolved": False,
        "model_or_checkpoint_opened": False,
        "cuda_initialized": False,
        "boundaries": {
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "normalizer_refit": False, "m30_rerun": False, "sealed_comparator_rerun": False,
        },
        "receipt_codec": "speed_smoke_v1__baseline_exact_optimized_numerical_equivalence",
    }


def base_plan_metric() -> Mapping[str, object]:
    """Import-free indirection keeps the standard last-bin metric literal explicit."""
    from src.causal_dual_memory_cell_d_score_v1 import plan as base_plan
    return base_plan.METRIC_CONTRACT


def _fixed_from_payload(value: object) -> shared_score.FixedEvaluationAuthority:
    try:
        return shared_score._fixed_authority_from_payload(value)
    except shared_score.ScoreError as error:
        raise SpeedSmokeError("speed-smoke fixed evaluation authority reconstruction drift") from error


def validate_target_free_preflight(value: object, identity: SpeedSmokeIdentity) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "source_gate", "completed_precision_v2", "evaluation_authority", "metric",
        "authority_root_relative", "score_root_relative", "canonical_launch_environment", "target_free",
        "target_paths_resolved", "model_or_checkpoint_opened", "cuda_initialized", "boundaries", "receipt_codec",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_target_free_preflight_v1"
        or value.get("status") != "PREFLIGHT_ACCEPTED" or value.get("identity") != _identity_payload(identity)
        or value.get("source_gate") != _identity_payload(identity)["completed_precision_v2_binding"]
        or value.get("completed_precision_v2") != _identity_payload(identity)["completed_precision_v2_binding"]
        or value.get("metric") != dict(base_plan_metric())
        or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE
        or value.get("canonical_launch_environment") != plan.validate_selected_launch_environment(
            {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS}
        )
        or value.get("target_free") is not True or value.get("target_paths_resolved") is not False
        or value.get("model_or_checkpoint_opened") is not False or value.get("cuda_initialized") is not False
        or value.get("boundaries") != {
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "normalizer_refit": False, "m30_rerun": False, "sealed_comparator_rerun": False,
        }
        or value.get("receipt_codec") != "speed_smoke_v1__baseline_exact_optimized_numerical_equivalence"
    ):
        raise SpeedSmokeError("speed-smoke preflight schema/identity/boundary drift")
    _fixed_from_payload(value.get("evaluation_authority"))
    return dict(value)


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    checked = dict(preflight)
    completed = checked.get("completed_precision_v2")
    if not isinstance(completed, Mapping):
        raise SpeedSmokeError("speed-smoke authorization completed V2 binding absent")
    return {
        "schema": "precision_aware_cdmd_speed_smoke_root_authorization_v1",
        "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": _sha(official_preflight_sha256, "speed-smoke preflight SHA"),
        "identity_sha256": _digest(checked["identity"]),
        "completed_precision_v2_binding_sha256": completed.get("binding_sha256"),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
        "gpu0_environment_required_before_reserve_publish_capability_attempt_and_final": True,
    }


def validate_root_authorization(
    value: object, *, official_preflight_sha256: str, preflight: Mapping[str, object], identity: SpeedSmokeIdentity,
) -> dict[str, object]:
    checked = validate_target_free_preflight(preflight, identity)
    expected = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=checked)
    if dict(value) != expected or expected["identity_sha256"] != identity.sha256:
        raise SpeedSmokeError("speed-smoke root authorization canonical identity drift")
    return expected


def _assert_current_closure(root: Path, identity: SpeedSmokeIdentity) -> None:
    if plan.implementation_closure(Path(root)).payload() != _identity_payload(identity)["closure"]:
        raise SpeedSmokeError("speed-smoke current implementation closure drift")


def _validate_mutating_launch_environment(environ: Mapping[str, str] | None) -> dict[str, object]:
    """Bind an issuer's claimed variables to the actual future process.

    A test mapping is useful for no-data validation, but it cannot authorize a
    reservation/publication if the inherited physical wrapper would later see
    a different process environment.  This remains lexical only: no source
    path resolve/stat/open occurs here.
    """
    checked = plan.validate_selected_launch_environment(environ)
    if environ is not None:
        keys = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", *plan.CANONICAL_SOURCE_ROOTS)
        if any(os.environ.get(key) != environ.get(key) for key in keys):
            raise SpeedSmokeError("speed-smoke supplied/actual launch environment drift")
    plan.validate_selected_launch_environment(os.environ)
    return checked


def reserve_authority_artifact(
    root: Path, capability: object, *, identity: SpeedSmokeIdentity, environ: Mapping[str, str] | None = None,
) -> shared_score.ArtifactRoot:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(capability)
    _require_live_completed_v2(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    plan.assert_fresh_prospective_root(Path(root), plan.AUTHORITY_ROOT_RELATIVE)
    parent = Path(root).absolute() / Path(plan.AUTHORITY_ROOT_RELATIVE).parent
    return shared_score._equal_session_module(Path(root)).reserve_artifact_root(
        parent, Path(plan.AUTHORITY_ROOT_RELATIVE).name, topology=AUTHORITY_TOPOLOGY,
    )


def publish_target_free_preflight(
    root: Path, artifact: shared_score.ArtifactRoot, capability: object, payload: Mapping[str, object], *,
    identity: SpeedSmokeIdentity, environ: Mapping[str, str] | None = None,
) -> str:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(capability)
    _require_live_completed_v2(Path(root), identity)
    checked = validate_target_free_preflight(payload, identity)
    observed_fixed = shared_score.derive_fixed_evaluation_authority(Path(root))
    if checked["evaluation_authority"] != observed_fixed.payload():
        raise SpeedSmokeError("speed-smoke fixed authority drift before preflight publication")
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    root: Path, artifact: shared_score.ArtifactRoot, capability: object, payload: Mapping[str, object], *,
    identity: SpeedSmokeIdentity, environ: Mapping[str, str] | None = None,
) -> str:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(capability)
    _require_live_completed_v2(Path(root), identity)
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise SpeedSmokeError("speed-smoke durable preflight malformed") from error
    if not isinstance(preflight, Mapping):
        raise SpeedSmokeError("speed-smoke durable preflight root drift")
    checked = validate_target_free_preflight(preflight, identity)
    observed_fixed = shared_score.derive_fixed_evaluation_authority(Path(root))
    if checked["evaluation_authority"] != observed_fixed.payload():
        raise SpeedSmokeError("speed-smoke fixed authority drift before authorization publication")
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=hashlib.sha256(body).hexdigest(), preflight=checked, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def _read_durable_authority_pair(
    root: Path, *, identity: SpeedSmokeIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    named = _directory_identity(directory, "speed-smoke durable authority")
    descriptor = -1
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _no_follow_flag())
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise SpeedSmokeError("speed-smoke durable authority identity drift before read")
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(descriptor)) != expected:
            raise SpeedSmokeError("speed-smoke durable authority exact topology drift")
        pre_body, preflight, pre_sha = shared_score._read_unbound_0444_pair(descriptor, "official_preflight.json")
        auth_body, authorization, auth_sha = shared_score._read_unbound_0444_pair(descriptor, "root_authorization.json")
        if _directory_identity(directory, "speed-smoke durable authority") != named:
            raise SpeedSmokeError("speed-smoke durable authority identity drift during read")
        validate_target_free_preflight(preflight, identity)
        validate_root_authorization(authorization, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)
        _require(hashlib.sha256(pre_body).hexdigest() == pre_sha and hashlib.sha256(auth_body).hexdigest() == auth_sha,
                 "speed-smoke durable authority body digest drift")
        return dict(preflight), dict(authorization), pre_sha, auth_sha
    except (OSError, shared_score.ScoreError) as error:
        raise SpeedSmokeError("speed-smoke durable authority descriptor read failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_durable_authority(
    root: Path, *, identity: SpeedSmokeIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    return _read_durable_authority_pair(Path(root), identity=identity)


def issue_durable_execution_capability(
    root: Path, *, identity: SpeedSmokeIdentity, root_capability: object,
    environ: Mapping[str, str] | None = None,
) -> shared_score.ExecutionCapability:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(root_capability)
    _require_live_completed_v2(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    observed_fixed = shared_score.derive_fixed_evaluation_authority(Path(root))
    if preflight["evaluation_authority"] != observed_fixed.payload():
        raise SpeedSmokeError("speed-smoke issuer fixed authority drift")
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    return shared_score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )


def reserve_score_artifact(
    root: Path, *, identity: SpeedSmokeIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> shared_score.ArtifactRoot:
    shared_score.require_execution_capability(capability, identity)
    _validate_mutating_launch_environment(environ)
    _require_live_completed_v2(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    approved = shared_score.require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise SpeedSmokeError("speed-smoke capability/durable authority drift")
    observed_fixed = shared_score.derive_fixed_evaluation_authority(Path(root))
    if preflight["evaluation_authority"] != observed_fixed.payload():
        raise SpeedSmokeError("speed-smoke score-reservation fixed authority drift")
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    parent = Path(root).absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return shared_score._equal_session_module(Path(root)).reserve_artifact_root(
        parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY,
    )


def _record_by_witness(records: Sequence[Mapping[str, object]], witness: plan.SmokeRowWitness) -> dict[str, object]:
    matches = [
        dict(record) for record in records
        if isinstance(record, Mapping) and record.get("surface") == witness.surface
        and record.get("session") == witness.session
    ]
    if len(matches) != 1:
        raise SpeedSmokeError("speed-smoke selected materialized input-record topology drift")
    return matches[0]


def input_payload(authority: Any, identity: SpeedSmokeIdentity) -> dict[str, object]:
    """Retain every materialized V2 input record while binding selected rows.

    The physical V1 substrate creates its canonical authority using the
    compatibility identity required by the exact parser.  This successor
    records that identity digest separately instead of lying that its new
    GPU0/result provenance is the historical V2 input-authority body.
    """
    runtime_identity = identity.runtime_identity()
    try:
        inherited = authority.payload(identity=runtime_identity)
    except (AttributeError, shared_score.ScoreError) as error:
        raise SpeedSmokeError("speed-smoke physical input authority reconstruction failed") from error
    records = inherited.get("records") if isinstance(inherited, Mapping) else None
    if not isinstance(records, list):
        raise SpeedSmokeError("speed-smoke physical input records are absent")
    historical_input = _identity_payload(identity)["completed_precision_v2_binding"].get("historical_v2_input_authority")
    if not isinstance(historical_input, Mapping) or historical_input.get("records") != records:
        raise SpeedSmokeError("speed-smoke full materialized input authority/V2 record equivalence drift")
    selected = {
        f"m{witness.budget}:{witness.surface}:{witness.session}": _digest(_record_by_witness(records, witness))
        for witness in plan.SMOKE_ROWS
    }
    expected = {
        f"m{witness.budget}:{witness.surface}:{witness.session}": witness.input_record_sha256
        for witness in plan.SMOKE_ROWS
    }
    if selected != expected:
        raise SpeedSmokeError("speed-smoke selected materialized input record/V2 witness drift")
    return {
        "schema": "precision_aware_cdmd_speed_smoke_input_authority_v1",
        "identity_sha256": identity.sha256,
        "runtime_v1_science_identity_sha256": _digest(runtime_identity.payload()),
        "historical_v2_input_authority_sha256": plan.V2_RESULT_SHAS["input_authority.json"],
        "full_historical_v2_input_record_equivalence": True,
        "fixed_evaluation_authority_sha256": inherited["fixed_evaluation_authority_sha256"],
        "records": records,
        "selected_record_sha256s": selected,
        "same_materialized_input_for_historical_v2_and_speed_smoke": True,
        "target_labels_metric_only": True,
        "cache_read_or_write": False,
    }


def validate_input_payload(
    value: object, *, identity: SpeedSmokeIdentity, evaluation_authority: shared_score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity_sha256", "runtime_v1_science_identity_sha256", "historical_v2_input_authority_sha256",
        "fixed_evaluation_authority_sha256", "records", "selected_record_sha256s",
        "same_materialized_input_for_historical_v2_and_speed_smoke", "full_historical_v2_input_record_equivalence",
        "target_labels_metric_only", "cache_read_or_write",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_input_authority_v1"
        or value.get("identity_sha256") != identity.sha256
        or value.get("runtime_v1_science_identity_sha256") != _digest(identity.runtime_identity().payload())
        or value.get("historical_v2_input_authority_sha256") != plan.V2_RESULT_SHAS["input_authority.json"]
        or value.get("fixed_evaluation_authority_sha256") != _digest(evaluation_authority.payload())
        or value.get("same_materialized_input_for_historical_v2_and_speed_smoke") is not True
        or value.get("full_historical_v2_input_record_equivalence") is not True
        or value.get("target_labels_metric_only") is not True or value.get("cache_read_or_write") is not False
        or not isinstance(value.get("records"), list) or not isinstance(value.get("selected_record_sha256s"), Mapping)
    ):
        raise SpeedSmokeError("speed-smoke input authority schema/boundary drift")
    records = [shared_score._input_record_from_payload(record).payload() for record in value["records"]]
    assets = (*evaluation_authority.within, *evaluation_authority.external)
    if len(records) != len(assets):
        raise SpeedSmokeError("speed-smoke input authority fixed roster cardinality drift")
    for record, asset in zip(records, assets, strict=True):
        if (
            record["surface"] != asset.surface or record["session"] != asset.session
            or record["asset_id"] != asset.asset_id or record["frozen_path"] != asset.frozen_path
            or record["asset_bytes"] != asset.bytes or record["asset_sha256"] != asset.sha256
        ):
            raise SpeedSmokeError("speed-smoke input authority immutable fixed-row drift")
    expected_selected = {
        f"m{witness.budget}:{witness.surface}:{witness.session}": _digest(_record_by_witness(records, witness))
        for witness in plan.SMOKE_ROWS
    }
    literal_selected = {
        f"m{witness.budget}:{witness.surface}:{witness.session}": witness.input_record_sha256
        for witness in plan.SMOKE_ROWS
    }
    if dict(value["selected_record_sha256s"]) != expected_selected or expected_selected != literal_selected:
        raise SpeedSmokeError("speed-smoke selected input-record digest drift")
    historical_input = _identity_payload(identity)["completed_precision_v2_binding"].get("historical_v2_input_authority")
    if not isinstance(historical_input, Mapping) or historical_input.get("records") != records:
        raise SpeedSmokeError("speed-smoke full V2 input record equivalence drift")
    rebuilt = {
        "schema": "precision_aware_cdmd_speed_smoke_input_authority_v1",
        "identity_sha256": identity.sha256,
        "runtime_v1_science_identity_sha256": _digest(identity.runtime_identity().payload()),
        "historical_v2_input_authority_sha256": plan.V2_RESULT_SHAS["input_authority.json"],
        "fixed_evaluation_authority_sha256": _digest(evaluation_authority.payload()),
        "records": records,
        "selected_record_sha256s": expected_selected,
        "same_materialized_input_for_historical_v2_and_speed_smoke": True,
        "full_historical_v2_input_record_equivalence": True,
        "target_labels_metric_only": True,
        "cache_read_or_write": False,
    }
    if dict(value) != rebuilt:
        raise SpeedSmokeError("speed-smoke input authority canonical reconstruction drift")
    return rebuilt


def _validate_speed_evidence(value: object) -> dict[str, object]:
    """Validate O1/O2 evidence plus the route-local physical-batch choice."""
    outer_required = {
        "schema", "stage0_o1_o2_engine_evidence", "physical_eval_batch_size",
        "physical_eval_batch_candidates", "fallback_attempts",
    }
    if (
        not isinstance(value, Mapping) or set(value) != outer_required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_o1_o2_evidence_v2"
        or value.get("physical_eval_batch_size") not in plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES
        or value.get("physical_eval_batch_candidates") != list(plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES)
        or not isinstance(value.get("fallback_attempts"), list) or not value["fallback_attempts"]
    ):
        raise SpeedSmokeError("speed-smoke optimized evidence schema/batch drift")
    attempts = value["fallback_attempts"]
    if (
        attempts[-1] != {"physical_eval_batch_size": value["physical_eval_batch_size"], "outcome": "selected"}
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"physical_eval_batch_size", "outcome"}
            or item.get("physical_eval_batch_size") not in plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES
            or item.get("outcome") not in {"selected", "cuda_oom_fallback"}
            for item in attempts
        )
        or [item["physical_eval_batch_size"] for item in attempts]
        != list(plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES[:len(attempts)])
        or any(item["outcome"] != "cuda_oom_fallback" for item in attempts[:-1])
    ):
        raise SpeedSmokeError("speed-smoke optimized OOM fallback evidence drift")
    engine = value.get("stage0_o1_o2_engine_evidence")
    required = {
        "schema", "logical_eval_batch_size", "identity_cache_entry_count", "identity_encoder_forward_count",
        "actual_model_forward_count", "actual_model_forward_count_by_path", "logical_chunk_count_by_path",
        "cache_key_semantics", "repeat_audit", "numeric_batch_variants_deferred",
    }
    if not isinstance(engine, Mapping) or set(engine) != required | {"physical_eval_batch_size"} or engine.get("schema") != "causal_dual_memory_cell_d_speed_v1_identity_cache_v1":
        raise SpeedSmokeError("speed-smoke evidence schema drift")
    if (
        engine.get("logical_eval_batch_size") != plan.LOGICAL_EVAL_BATCH_SIZE
        or engine.get("cache_key_semantics") != "path_plus_contiguous_activity_prefix_normalized_t4_and_held_view_bytes"
        or engine.get("numeric_batch_variants_deferred") != [1024, 2048]
        or engine.get("physical_eval_batch_size") != value["physical_eval_batch_size"]
    ):
        raise SpeedSmokeError("speed-smoke evidence static contract drift")
    for key in ("identity_cache_entry_count", "identity_encoder_forward_count", "actual_model_forward_count"):
        if type(engine.get(key)) is not int or engine[key] <= 0:
            raise SpeedSmokeError(f"speed-smoke evidence {key} drift")
    by_path = engine.get("actual_model_forward_count_by_path")
    logical = engine.get("logical_chunk_count_by_path")
    repeat = engine.get("repeat_audit")
    if (
        not isinstance(by_path, Mapping) or not isinstance(logical, Mapping) or not isinstance(repeat, Mapping)
        or sum(item for item in by_path.values() if type(item) is int) != engine["actual_model_forward_count"]
        or any(type(item) is not int or item <= 0 for item in by_path.values())
        or any(type(item) is not int or item <= 0 for item in logical.values())
        or repeat.get("schema") != "causal_dual_memory_cell_d_speed_v1_repeat_audit_v1"
        or repeat.get("canonical_full_first_chunk_per_exact_state") is not True
        or repeat.get("held_group_0_fixed_mid_session_first_chunk") is not True
        or repeat.get("repeat_audit_uses_no_target_values") is not True
        or not isinstance(repeat.get("coordinates"), list)
    ):
        raise SpeedSmokeError("speed-smoke evidence forward/repeat topology drift")
    coordinates = repeat["coordinates"]
    if not coordinates or any(not isinstance(item, Mapping) or item.get("outputs_bitwise_equal") is not True for item in coordinates):
        raise SpeedSmokeError("speed-smoke evidence repeated-forward parity drift")
    # The full path is always repeated once per exact post-transition state;
    # group-0 has exactly one fixed mid-session coverage coordinate.
    if not any(item.get("path") == "full" and "canonical_full_first_state" in item.get("reasons", []) for item in coordinates):
        raise SpeedSmokeError("speed-smoke evidence canonical full repeat absent")
    if not any(item.get("path") == "held_group_0" and "held_group_0_fixed_mid_session" in item.get("reasons", []) for item in coordinates):
        raise SpeedSmokeError("speed-smoke evidence held group repeat absent")
    return dict(value)


def _validate_resources(value: object, *, identity: SpeedSmokeIdentity) -> dict[str, object]:
    required = {
        "runtime_environment", "current_cuda_allocated_bytes", "current_cuda_reserved_bytes",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "rss_bytes", "wall_seconds",
        "full_and_group_forward_chunks", "completed_query_trials", "windows_or_trials_per_s",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise SpeedSmokeError("speed-smoke resources schema drift")
    expected_environment = {
        **identity.payload()["selected_device_profile"], "visible_devices": 1, "attested": True,
        "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False,
    }
    if value.get("runtime_environment") != expected_environment:
        raise SpeedSmokeError("speed-smoke resources selected-device/TF32 drift")
    integers = (
        "current_cuda_allocated_bytes", "current_cuda_reserved_bytes", "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes", "rss_bytes", "full_and_group_forward_chunks", "completed_query_trials",
    )
    if any(type(value.get(key)) is not int or value[key] < 0 for key in integers):
        raise SpeedSmokeError("speed-smoke resources integer drift")
    if (
        value["peak_cuda_allocated_bytes"] < value["current_cuda_allocated_bytes"]
        or value["peak_cuda_reserved_bytes"] < value["current_cuda_reserved_bytes"]
        or value["rss_bytes"] <= 0 or value["full_and_group_forward_chunks"] <= 0
        or value["completed_query_trials"] <= 0
    ):
        raise SpeedSmokeError("speed-smoke resources measured counter/memory drift")
    for key in ("wall_seconds", "windows_or_trials_per_s"):
        item = value.get(key)
        if type(item) is not float or not (item > 0.0) or item != item or item in {float("inf"), float("-inf")}:
            raise SpeedSmokeError(f"speed-smoke resources {key} finiteness drift")
    return dict(value)


def _witness_for(budget: int, surface: str, session: str) -> plan.SmokeRowWitness:
    matches = [
        item for item in plan.SMOKE_ROWS
        if item.budget == budget and item.surface == surface and item.session == session
    ]
    if len(matches) != 1:
        raise SpeedSmokeError("speed-smoke cell is outside fixed one-row matrix")
    return matches[0]


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SpeedSmokeError(f"speed-smoke {label} must be a finite float")
    return float(value)


def _validated_transition_records(row: Mapping[str, object], witness: plan.SmokeRowWitness) -> list[dict[str, object]]:
    transitions = row.get("transition_records")
    if not isinstance(transitions, list) or len(transitions) != witness.transition_count:
        raise SpeedSmokeError("speed-smoke transition evidence topology drift")
    if any(not isinstance(item, Mapping) for item in transitions):
        raise SpeedSmokeError("speed-smoke transition row type drift")
    checked = [dict(item) for item in transitions]
    if sum(item.get("carrier_transition_committed") is True for item in checked) != witness.carrier_commits:
        raise SpeedSmokeError("speed-smoke carrier-transition count drift")
    return checked


def _validate_baseline_row(row: Mapping[str, object], witness: plan.SmokeRowWitness) -> tuple[dict[str, object], list[dict[str, object]]]:
    checked = dict(row)
    if (
        _digest(checked) != witness.canonical_row_sha256
        or checked.get("session") != witness.session or checked.get("budget") != witness.budget
        or checked.get("governing_r2") != witness.governing_r2
        or checked.get("prediction_sha256") != witness.prediction_sha256
        or checked.get("input_record_sha256") != witness.input_record_sha256
        or checked.get("n_windows") != witness.n_windows
        or checked.get("model_state_before_sha256") != witness.model_state_sha256
        or checked.get("model_state_after_sha256") != witness.model_state_sha256
        or checked.get("sealed_model_load_proof_sha256") != witness.sealed_model_load_proof_sha256
    ):
        raise SpeedSmokeError("speed-smoke eager baseline historical-row parity drift")
    return checked, _validated_transition_records(checked, witness)


def _validate_optimized_row(
    row: Mapping[str, object], witness: plan.SmokeRowWitness, baseline_transitions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    checked = dict(row)
    runtime_r2 = _finite_float(checked.get("governing_r2"), "optimized R2")
    if (
        checked.get("session") != witness.session or checked.get("budget") != witness.budget
        or checked.get("input_record_sha256") != witness.input_record_sha256
        or checked.get("n_windows") != witness.n_windows
        or checked.get("model_state_before_sha256") != witness.model_state_sha256
        or checked.get("model_state_after_sha256") != witness.model_state_sha256
        or checked.get("sealed_model_load_proof_sha256") != witness.sealed_model_load_proof_sha256
        or _sha(checked.get("prediction_sha256"), "optimized prediction") != checked.get("prediction_sha256")
        or abs(runtime_r2 - witness.governing_r2) > plan.R2_ABSOLUTE_TOLERANCE
    ):
        raise SpeedSmokeError("speed-smoke optimized row static/R2 parity drift")
    transitions = _validated_transition_records(checked, witness)
    if transitions != [dict(item) for item in baseline_transitions]:
        raise SpeedSmokeError("speed-smoke optimized transition chronology drift")
    return checked, transitions


def _validate_numerical_comparison(
    value: object, *, baseline: Mapping[str, object], optimized: Mapping[str, object],
    baseline_transitions: Sequence[Mapping[str, object]], optimized_transitions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    required = {
        "schema", "baseline_physical_eval_batch_size", "optimized_physical_eval_batch_size",
        "baseline_prediction_sha256", "optimized_prediction_sha256", "max_abs_prediction_error",
        "prediction_max_abs_tolerance", "baseline_governing_r2", "optimized_governing_r2", "r2_abs_error",
        "r2_absolute_tolerance", "baseline_transition_records_sha256", "optimized_transition_records_sha256",
        "transition_sequence_exact", "baseline_wall_seconds", "optimized_wall_seconds", "speedup_ratio",
        "baseline_full_system_forward_count", "baseline_group_forward_count",
        "optimized_full_system_forward_count", "optimized_group_forward_count",
        "smoke_min_speedup_ratio_exclusive", "recommended_speedup_ratio",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "precision_aware_cdmd_speed_smoke_numerical_comparison_v1":
        raise SpeedSmokeError("speed-smoke numerical-comparison schema drift")
    expected_transitions = _digest([dict(item) for item in baseline_transitions])
    if (
        value.get("baseline_physical_eval_batch_size") != plan.BASELINE_PHYSICAL_EVAL_BATCH_SIZE
        or value.get("optimized_physical_eval_batch_size") not in plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES
        or value.get("baseline_prediction_sha256") != baseline.get("prediction_sha256")
        or value.get("optimized_prediction_sha256") != optimized.get("prediction_sha256")
        or value.get("baseline_governing_r2") != baseline.get("governing_r2")
        or value.get("optimized_governing_r2") != optimized.get("governing_r2")
        or value.get("baseline_transition_records_sha256") != expected_transitions
        or value.get("optimized_transition_records_sha256") != _digest([dict(item) for item in optimized_transitions])
        or value.get("transition_sequence_exact") is not True
        or value.get("prediction_max_abs_tolerance") != plan.PREDICTION_MAX_ABSOLUTE_TOLERANCE
        or value.get("r2_absolute_tolerance") != plan.R2_ABSOLUTE_TOLERANCE
        or value.get("baseline_full_system_forward_count") != baseline.get("full_system_forward_count")
        or value.get("baseline_group_forward_count") != baseline.get("group_forward_count")
        or value.get("optimized_full_system_forward_count") != optimized.get("full_system_forward_count")
        or value.get("optimized_group_forward_count") != optimized.get("group_forward_count")
        or value.get("smoke_min_speedup_ratio_exclusive") != plan.SMOKE_MIN_SPEEDUP_RATIO
        or value.get("recommended_speedup_ratio") != plan.RECOMMENDED_SPEEDUP_RATIO
    ):
        raise SpeedSmokeError("speed-smoke numerical-comparison literal/transition drift")
    max_abs = _finite_float(value.get("max_abs_prediction_error"), "max absolute prediction error")
    r2_error = _finite_float(value.get("r2_abs_error"), "R2 absolute error")
    baseline_wall = _finite_float(value.get("baseline_wall_seconds"), "baseline wall seconds")
    optimized_wall = _finite_float(value.get("optimized_wall_seconds"), "optimized wall seconds")
    speedup = _finite_float(value.get("speedup_ratio"), "speedup ratio")
    if (
        max_abs < 0.0 or max_abs > plan.PREDICTION_MAX_ABSOLUTE_TOLERANCE
        or r2_error != abs(float(baseline["governing_r2"]) - float(optimized["governing_r2"]))
        or r2_error > plan.R2_ABSOLUTE_TOLERANCE
        or baseline_wall <= 0.0 or optimized_wall <= 0.0
        or speedup != baseline_wall / optimized_wall or speedup <= plan.SMOKE_MIN_SPEEDUP_RATIO
        or any(
            type(value.get(key)) is not int or value[key] < 0
            for key in (
                "baseline_full_system_forward_count", "baseline_group_forward_count",
                "optimized_full_system_forward_count", "optimized_group_forward_count",
            )
        )
    ):
        raise SpeedSmokeError("speed-smoke numerical-comparison tolerance/speedup drift")
    return dict(value)


def build_speed_cell(
    *, identity: SpeedSmokeIdentity, input_authority_sha256: str,
    baseline_session_row: Mapping[str, object], optimized_session_row: Mapping[str, object],
    comparison: Mapping[str, object], speed_evidence: Mapping[str, object], resources: Mapping[str, object],
    witness: plan.SmokeRowWitness,
) -> dict[str, object]:
    """Build one baseline-anchored, numerically equivalent speed-cell receipt."""
    baseline, baseline_transitions = _validate_baseline_row(baseline_session_row, witness)
    optimized, optimized_transitions = _validate_optimized_row(
        optimized_session_row, witness, baseline_transitions,
    )
    checked_comparison = _validate_numerical_comparison(
        comparison, baseline=baseline, optimized=optimized,
        baseline_transitions=baseline_transitions, optimized_transitions=optimized_transitions,
    )
    evidence = _validate_speed_evidence(speed_evidence)
    checked_resources = _validate_resources(resources, identity=identity)
    engine = evidence["stage0_o1_o2_engine_evidence"]
    forward = {
        "logical_eval_batch_size": plan.LOGICAL_EVAL_BATCH_SIZE,
        "optimized_physical_eval_batch_size": evidence["physical_eval_batch_size"],
        "optimized_actual_model_forward_count": engine["actual_model_forward_count"],
        "optimized_identity_encoder_forward_count": engine["identity_encoder_forward_count"],
        "optimized_actual_model_forward_count_by_path": engine["actual_model_forward_count_by_path"],
        "optimized_logical_chunk_count_by_path": engine["logical_chunk_count_by_path"],
        "baseline_full_system_forward_count": baseline["full_system_forward_count"],
        "baseline_group_forward_count": baseline["group_forward_count"],
        "optimized_full_system_forward_count": optimized["full_system_forward_count"],
        "optimized_group_forward_count": optimized["group_forward_count"],
    }
    return {
        "schema": "precision_aware_cdmd_speed_smoke_cell_v1",
        "budget": witness.budget,
        "surface": witness.surface,
        "session": witness.session,
        "historical_v2_row_sha256": witness.canonical_row_sha256,
        "input_authority_sha256": _sha(input_authority_sha256, "speed-smoke cell input authority"),
        "historical_input_record_sha256": witness.input_record_sha256,
        "runtime_input_record_sha256": optimized["input_record_sha256"],
        "historical_governing_r2": witness.governing_r2,
        "baseline_governing_r2": baseline["governing_r2"],
        "optimized_governing_r2": optimized["governing_r2"],
        "historical_prediction_sha256": witness.prediction_sha256,
        "baseline_prediction_sha256": baseline["prediction_sha256"],
        "optimized_prediction_sha256": optimized["prediction_sha256"],
        "baseline_historical_row_exact": True,
        "baseline_row_sha256": _digest(baseline),
        "optimized_row_sha256": _digest(optimized),
        "n_windows": witness.n_windows,
        "carrier_transition_committed_count": witness.carrier_commits,
        "transition_count": witness.transition_count,
        "model_state_before_sha256": witness.model_state_sha256,
        "model_state_after_sha256": witness.model_state_sha256,
        "sealed_model_load_proof_sha256": witness.sealed_model_load_proof_sha256,
        "speed_evidence": evidence,
        "forward_counts": forward,
        "resources": checked_resources,
        "numerical_equivalence": checked_comparison,
        "metric": "last_bin_variance_weighted_two_output_r2",
        "baseline_exact_prediction_sha256_parity": True,
        "optimized_prediction_numerically_equivalent": True,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "normalizer_refit": False,
        "m30_rerun": False,
        "sealed_comparator_rerun": False,
    }


def validate_speed_cell(
    value: object, *, identity: SpeedSmokeIdentity, input_payload_value: Mapping[str, object],
    input_authority_sha256: str, witness: plan.SmokeRowWitness,
) -> dict[str, object]:
    required = {
        "schema", "budget", "surface", "session", "historical_v2_row_sha256", "input_authority_sha256",
        "historical_input_record_sha256", "runtime_input_record_sha256", "historical_governing_r2",
        "baseline_governing_r2", "optimized_governing_r2", "historical_prediction_sha256",
        "baseline_prediction_sha256", "optimized_prediction_sha256", "baseline_historical_row_exact",
        "baseline_row_sha256", "optimized_row_sha256", "n_windows", "carrier_transition_committed_count",
        "transition_count", "model_state_before_sha256", "model_state_after_sha256", "sealed_model_load_proof_sha256",
        "speed_evidence", "forward_counts", "resources", "numerical_equivalence", "metric",
        "baseline_exact_prediction_sha256_parity", "optimized_prediction_numerically_equivalent",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "normalizer_refit", "m30_rerun", "sealed_comparator_rerun",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "precision_aware_cdmd_speed_smoke_cell_v1":
        raise SpeedSmokeError("speed-smoke cell schema drift")
    if (
        value.get("budget") != witness.budget or value.get("surface") != witness.surface
        or value.get("session") != witness.session or value.get("historical_v2_row_sha256") != witness.canonical_row_sha256
        or value.get("input_authority_sha256") != _sha(input_authority_sha256, "speed-smoke expected input SHA")
        or value.get("historical_input_record_sha256") != witness.input_record_sha256
        or value.get("runtime_input_record_sha256") != witness.input_record_sha256
        or value.get("historical_governing_r2") != witness.governing_r2
        or value.get("baseline_governing_r2") != witness.governing_r2
        or not isinstance(value.get("optimized_governing_r2"), float)
        or value.get("historical_prediction_sha256") != witness.prediction_sha256
        or value.get("baseline_prediction_sha256") != witness.prediction_sha256
        or value.get("baseline_historical_row_exact") is not True
        or value.get("baseline_row_sha256") != witness.canonical_row_sha256
        or _sha(value.get("optimized_prediction_sha256"), "speed-smoke optimized receipt prediction") != value.get("optimized_prediction_sha256")
        or _sha(value.get("optimized_row_sha256"), "speed-smoke optimized receipt row") != value.get("optimized_row_sha256")
        or value.get("n_windows") != witness.n_windows
        or value.get("carrier_transition_committed_count") != witness.carrier_commits
        or value.get("transition_count") != witness.transition_count
        or value.get("model_state_before_sha256") != witness.model_state_sha256
        or value.get("model_state_after_sha256") != witness.model_state_sha256
        or value.get("sealed_model_load_proof_sha256") != witness.sealed_model_load_proof_sha256
        or value.get("metric") != "last_bin_variance_weighted_two_output_r2"
        or value.get("baseline_exact_prediction_sha256_parity") is not True
        or value.get("optimized_prediction_numerically_equivalent") is not True
        or any(value.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls"))
        or value.get("normalizer_refit") is not False or value.get("m30_rerun") is not False
        or value.get("sealed_comparator_rerun") is not False
    ):
        raise SpeedSmokeError("speed-smoke baseline/optimized cell boundary drift")
    evidence = _validate_speed_evidence(value.get("speed_evidence"))
    engine = evidence["stage0_o1_o2_engine_evidence"]
    comparison = value.get("numerical_equivalence")
    if not isinstance(comparison, Mapping):
        raise SpeedSmokeError("speed-smoke receipt numerical-comparison type drift")
    historical_rows = _identity_payload(identity)["completed_precision_v2_binding"].get("historical_selected_rows")
    if not isinstance(historical_rows, list):
        raise SpeedSmokeError("speed-smoke completed V2 selected rows are absent")
    historical = [dict(item) for item in historical_rows if isinstance(item, Mapping) and _digest(dict(item)) == witness.canonical_row_sha256]
    if len(historical) != 1:
        raise SpeedSmokeError("speed-smoke completed V2 baseline-row witness drift")
    historical_transitions = _validated_transition_records(historical[0], witness)
    optimized_stub = {
        "prediction_sha256": value["optimized_prediction_sha256"],
        "governing_r2": value["optimized_governing_r2"],
        "full_system_forward_count": comparison.get("optimized_full_system_forward_count"),
        "group_forward_count": comparison.get("optimized_group_forward_count"),
    }
    _validate_numerical_comparison(
        comparison, baseline=historical[0], optimized=optimized_stub,
        baseline_transitions=historical_transitions, optimized_transitions=historical_transitions,
    )
    forward = value.get("forward_counts")
    expected_forward = {
        "logical_eval_batch_size": plan.LOGICAL_EVAL_BATCH_SIZE,
        "optimized_physical_eval_batch_size": evidence["physical_eval_batch_size"],
        "optimized_actual_model_forward_count": engine["actual_model_forward_count"],
        "optimized_identity_encoder_forward_count": engine["identity_encoder_forward_count"],
        "optimized_actual_model_forward_count_by_path": engine["actual_model_forward_count_by_path"],
        "optimized_logical_chunk_count_by_path": engine["logical_chunk_count_by_path"],
        "baseline_full_system_forward_count": comparison["baseline_full_system_forward_count"],
        "baseline_group_forward_count": comparison["baseline_group_forward_count"],
        "optimized_full_system_forward_count": comparison["optimized_full_system_forward_count"],
        "optimized_group_forward_count": comparison["optimized_group_forward_count"],
    }
    if forward != expected_forward:
        raise SpeedSmokeError("speed-smoke forward-count evidence drift")
    _validate_resources(value.get("resources"), identity=identity)
    records = input_payload_value.get("records")
    if not isinstance(records, list) or _digest(_record_by_witness(records, witness)) != witness.input_record_sha256:
        raise SpeedSmokeError("speed-smoke cell/runtime input record drift")
    return dict(value)


def summarize_budget(
    cells: Sequence[Mapping[str, object]], *, budget: int, input_payload_value: Mapping[str, object],
    identity: SpeedSmokeIdentity, input_authority_sha256: str,
) -> dict[str, object]:
    expected = [item for item in plan.SMOKE_ROWS if item.budget == budget]
    if len(expected) != 1 or len(cells) != 1:
        raise SpeedSmokeError("speed-smoke budget cell cardinality drift")
    checked = validate_speed_cell(
        cells[0], identity=identity, input_payload_value=input_payload_value,
        input_authority_sha256=input_authority_sha256, witness=expected[0],
    )
    return {
        "schema": "precision_aware_cdmd_speed_smoke_budget_summary_v1",
        "budget": budget,
        "cells": [checked],
        "baseline_historical_precision_v2_parity": True,
        "non_governing_engineering_smoke": True,
    }


def budget_gate(
    summary: Mapping[str, object], *, budget: int, identity: SpeedSmokeIdentity, input_payload_value: Mapping[str, object],
    input_authority_sha256: str,
) -> dict[str, object]:
    expected = summarize_budget(
        summary.get("cells", ()) if isinstance(summary, Mapping) else (), budget=budget,
        input_payload_value=input_payload_value, identity=identity, input_authority_sha256=input_authority_sha256,
    )
    if dict(summary) != expected:
        raise SpeedSmokeError("speed-smoke budget summary reconstruction drift")
    return {
        "budget": budget,
        "baseline_exact_prediction_sha256_parity": True,
        "optimized_numerical_equivalence": True,
        "completed": True,
        "formal_gate": False,
    }


def _score_payload(
    identity: SpeedSmokeIdentity, input_sha: str, summaries: Mapping[str, object], gates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    expected_keys = {str(item.budget) for item in plan.SMOKE_ROWS}
    if set(summaries) != expected_keys or set(gates) != {item.budget for item in plan.SMOKE_ROWS}:
        raise SpeedSmokeError("speed-smoke score budget topology drift")
    payload = _identity_payload(identity)
    return {
        "schema": "precision_aware_cdmd_speed_smoke_score_v1",
        "identity": payload,
        "input_authority_sha256": _sha(input_sha, "speed-smoke score input authority"),
        "budget_summaries": {str(item.budget): dict(summaries[str(item.budget)]) for item in plan.SMOKE_ROWS},
        "budget_gates": {str(item.budget): dict(gates[item.budget]) for item in plan.SMOKE_ROWS},
        "cell_execution_order": [
            {"budget": budget, "surface": surface, "session": session}
            for budget, surface, session in plan.SMOKE_EXECUTION_ORDER
        ],
        "historical_v2_binding_sha256": payload["completed_precision_v2_binding_sha256"],
        "historical_v2_m30_reference_only": True,
        "sealed_comparator_rerun": False,
        "target_optimizer_backward_update": 0,
        "formal_verdict_emitted": False,
    }


def validate_score_payload(
    value: object, *, identity: SpeedSmokeIdentity, input_payload_value: Mapping[str, object],
    input_authority_sha256: str, evaluation_authority: shared_score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity", "input_authority_sha256", "budget_summaries", "budget_gates", "cell_execution_order",
        "historical_v2_binding_sha256", "historical_v2_m30_reference_only", "sealed_comparator_rerun",
        "target_optimizer_backward_update", "formal_verdict_emitted",
    }
    expected_input = _sha(input_authority_sha256, "speed-smoke expected score input SHA")
    if _digest(input_payload_value) != expected_input:
        raise SpeedSmokeError("speed-smoke score input body/SHA drift")
    validate_input_payload(input_payload_value, identity=identity, evaluation_authority=evaluation_authority)
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_score_v1"
        or value.get("identity") != _identity_payload(identity) or value.get("input_authority_sha256") != expected_input
        or value.get("cell_execution_order") != [
            {"budget": budget, "surface": surface, "session": session}
            for budget, surface, session in plan.SMOKE_EXECUTION_ORDER
        ]
        or value.get("historical_v2_binding_sha256") != _identity_payload(identity)["completed_precision_v2_binding_sha256"]
        or value.get("historical_v2_m30_reference_only") is not True
        or value.get("sealed_comparator_rerun") is not False
        or value.get("target_optimizer_backward_update") != 0 or value.get("formal_verdict_emitted") is not False
        or not isinstance(value.get("budget_summaries"), Mapping) or not isinstance(value.get("budget_gates"), Mapping)
    ):
        raise SpeedSmokeError("speed-smoke score schema/identity/matrix drift")
    summaries = value["budget_summaries"]
    gates = value["budget_gates"]
    expected_budgets = {str(item.budget) for item in plan.SMOKE_ROWS}
    if set(summaries) != expected_budgets or set(gates) != expected_budgets:
        raise SpeedSmokeError("speed-smoke score exact one-budget topology drift")
    rebuilt_summaries: dict[str, object] = {}
    rebuilt_gates: dict[str, object] = {}
    for witness in plan.SMOKE_ROWS:
        raw = summaries[str(witness.budget)]
        if not isinstance(raw, Mapping):
            raise SpeedSmokeError("speed-smoke score budget summary type drift")
        rebuilt = summarize_budget(
            raw.get("cells", ()), budget=witness.budget, input_payload_value=input_payload_value,
            identity=identity, input_authority_sha256=expected_input,
        )
        if dict(raw) != rebuilt:
            raise SpeedSmokeError("speed-smoke score budget summary canonical drift")
        gate = budget_gate(
            rebuilt, budget=witness.budget, identity=identity, input_payload_value=input_payload_value,
            input_authority_sha256=expected_input,
        )
        if gates.get(str(witness.budget)) != gate:
            raise SpeedSmokeError("speed-smoke score budget gate drift")
        rebuilt_summaries[str(witness.budget)] = rebuilt
        rebuilt_gates[str(witness.budget)] = gate
    return dict(value)


def terminal_verdict(gates: Mapping[int, Mapping[str, object]]) -> str:
    if set(gates) != {item.budget for item in plan.SMOKE_ROWS} or any(item.get("completed") is not True for item in gates.values()):
        raise SpeedSmokeError("speed-smoke terminal requires exact completed one-cell matrix")
    return "SPEED_SMOKE_NUMERICAL_EQUIVALENCE_COMPLETE_NON_GOVERNING"


def _terminal_payload(
    identity: SpeedSmokeIdentity, attempt_sha: str, input_sha: str, score_sha: str,
    gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> dict[str, object]:
    if verdict != "SPEED_SMOKE_NUMERICAL_EQUIVALENCE_COMPLETE_NON_GOVERNING":
        raise SpeedSmokeError("speed-smoke terminal verdict drift")
    closure_sha = _identity_payload(identity)["closure"]["closure_sha256"]
    return {
        "schema": "precision_aware_cdmd_speed_smoke_terminal_v1",
        "status": "TERMINAL",
        "verdict": verdict,
        "identity": _identity_payload(identity),
        "attempt_sha256": _sha(attempt_sha, "speed-smoke terminal attempt"),
        "input_authority_sha256": _sha(input_sha, "speed-smoke terminal input"),
        "score_sha256": _sha(score_sha, "speed-smoke terminal score"),
        "budget_gates": {str(item.budget): dict(gates[item.budget]) for item in plan.SMOKE_ROWS},
        "launch_closure_sha256": closure_sha,
        "final_closure_sha256": closure_sha,
        "completed_precision_v2_binding_sha256": _identity_payload(identity)["completed_precision_v2_binding_sha256"],
        "target_optimizer_backward_update": 0,
        "m30_reference_only": True,
        "sealed_comparator_rerun": False,
    }


def validate_terminal_payload(
    value: object, *, identity: SpeedSmokeIdentity, attempt_sha256: str, input_authority_sha256: str,
    score_payload: Mapping[str, object], score_sha256: str,
) -> dict[str, object]:
    gates = score_payload.get("budget_gates") if isinstance(score_payload, Mapping) else None
    if not isinstance(gates, Mapping):
        raise SpeedSmokeError("speed-smoke terminal score gates absent")
    typed = {int(key): item for key, item in gates.items() if isinstance(key, str) and key.isdigit() and isinstance(item, Mapping)}
    expected = _terminal_payload(
        identity, attempt_sha256, input_authority_sha256, score_sha256, typed, terminal_verdict(typed),
    )
    if dict(value) != expected:
        raise SpeedSmokeError("speed-smoke terminal canonical graph drift")
    return expected


def _attempt_payload(identity: SpeedSmokeIdentity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    return {
        "schema": "precision_aware_cdmd_speed_smoke_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "identity": _identity_payload(identity),
        "preflight_sha256": _sha(pre_sha, "speed-smoke attempt preflight"),
        "authorization_sha256": _sha(auth_sha, "speed-smoke attempt authorization"),
        "target_paths_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "attempt_precedes_backend_prepare": True,
    }


def _failure_progress(value: object | None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "within_assets_opened": False, "external_assets_opened": False, "checkpoint_opened": False,
        "cuda_initialized": False, "full_system_forward_count": 0, "group_forward_count": 0,
    }
    if value is None:
        return defaults
    if not isinstance(value, Mapping) or set(value) != set(defaults):
        raise SpeedSmokeError("speed-smoke failure progress schema drift")
    result = dict(value)
    if any(type(result[key]) is not bool for key in ("within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized")):
        raise SpeedSmokeError("speed-smoke failure progress boolean drift")
    if any(type(result[key]) is not int or result[key] < 0 for key in ("full_system_forward_count", "group_forward_count")):
        raise SpeedSmokeError("speed-smoke failure progress counter drift")
    return result


def _failure_payload(
    identity: SpeedSmokeIdentity, attempt_sha: str, input_sha: str | None, stage: str,
    error: BaseException, runtime_progress: object | None,
) -> dict[str, object]:
    progress = _failure_progress(runtime_progress)
    return {
        "schema": "precision_aware_cdmd_speed_smoke_failure_v1",
        "status": "FAILED",
        "identity": _identity_payload(identity),
        "attempt_sha256": _sha(attempt_sha, "speed-smoke failure attempt"),
        "input_authority_sha256": None if input_sha is None else _sha(input_sha, "speed-smoke failure input"),
        "stage": stage,
        "error_class": type(error).__name__,
        "error_sha256": hashlib.sha256(repr(error).encode("utf-8")).hexdigest(),
        "target_paths_resolved_or_opened": bool(progress["within_assets_opened"] or progress["external_assets_opened"]),
        **progress,
        "terminal_published": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "completed_precision_v2_binding_sha256": _identity_payload(identity)["completed_precision_v2_binding_sha256"],
    }


def validate_failure_payload(
    value: object, *, identity: SpeedSmokeIdentity, attempt_sha256: str, input_authority_sha256: str | None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage", "error_class",
        "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened", "external_assets_opened",
        "checkpoint_opened", "cuda_initialized", "full_system_forward_count", "group_forward_count", "terminal_published",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "completed_precision_v2_binding_sha256",
    }
    stages = {"attempt", "prepare", "materialize_inputs", "budget_m4", "final_revalidate", "publish_terminal"}
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_failure_v1" or value.get("status") != "FAILED"
        or value.get("identity") != _identity_payload(identity) or value.get("attempt_sha256") != attempt_sha256
        or value.get("input_authority_sha256") != input_authority_sha256 or value.get("stage") not in stages
        or not isinstance(value.get("error_class"), str) or not value["error_class"]
        or not isinstance(value.get("error_sha256"), str) or value.get("terminal_published") is not False
        or any(value.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls"))
        or value.get("completed_precision_v2_binding_sha256") != _identity_payload(identity)["completed_precision_v2_binding_sha256"]
    ):
        raise SpeedSmokeError("speed-smoke failure schema/provenance drift")
    progress = _failure_progress({key: value.get(key) for key in (
        "within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized",
        "full_system_forward_count", "group_forward_count",
    )})
    if value.get("target_paths_resolved_or_opened") is not bool(progress["within_assets_opened"] or progress["external_assets_opened"]):
        raise SpeedSmokeError("speed-smoke failure target/progress drift")
    return dict(value)


def _validate_preflight_hook(value: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle identity type drift")
    return validate_target_free_preflight(value, identity)


def _validate_authorization_hook(
    value: Mapping[str, object], pre_sha: str, preflight: Mapping[str, object], identity: Any,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle authorization identity type drift")
    return validate_root_authorization(value, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)


def _closure_hook(root: Path) -> Mapping[str, object]:
    return plan.implementation_closure(Path(root)).payload()


def _source_gate_hook(root: Path) -> CompletedV2Binding:
    return validate_completed_precision_v2(Path(root))


def _fresh_score_hook(root: Path) -> None:
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)


def _fixed_from_preflight(preflight: Mapping[str, object]) -> shared_score.FixedEvaluationAuthority:
    return _fixed_from_payload(preflight.get("evaluation_authority"))


def _input_payload_hook(authority: Any, identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle input identity type drift")
    return input_payload(authority, identity)


def _validate_input_hook(value: Mapping[str, object], identity: Any, fixed: Any) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity) or not isinstance(fixed, shared_score.FixedEvaluationAuthority):
        raise SpeedSmokeError("speed-smoke lifecycle input typed boundary drift")
    return validate_input_payload(value, identity=identity, evaluation_authority=fixed)


def _summarize_hook(cells: Sequence[Mapping[str, object]], budget: int, input_payload_value: Mapping[str, object]) -> Mapping[str, object]:
    # The shared engine intentionally has no identity/input SHA argument at
    # this hook.  The backend returns already self-authenticated cell receipts;
    # final score validation repeats the full identity/input reconstruction.
    expected = [item for item in plan.SMOKE_ROWS if item.budget == budget]
    if len(expected) != 1 or len(cells) != 1 or not isinstance(cells[0], Mapping):
        raise SpeedSmokeError("speed-smoke lifecycle bounded cell topology drift")
    cell = dict(cells[0])
    return {
        "schema": "precision_aware_cdmd_speed_smoke_budget_summary_v1",
        "budget": budget,
        "cells": [cell],
        "baseline_historical_precision_v2_parity": True,
        "non_governing_engineering_smoke": True,
    }


def _budget_gate_hook(summary: Mapping[str, object], budget: int) -> Mapping[str, object]:
    if (
        not isinstance(summary, Mapping) or summary.get("budget") != budget
        or summary.get("baseline_historical_precision_v2_parity") is not True
        or summary.get("non_governing_engineering_smoke") is not True
        or not isinstance(summary.get("cells"), list) or len(summary["cells"]) != 1
    ):
        raise SpeedSmokeError("speed-smoke lifecycle budget summary gate drift")
    return {
        "budget": budget,
        "baseline_exact_prediction_sha256_parity": True,
        "optimized_numerical_equivalence": True,
        "completed": True,
        "formal_gate": False,
    }


def _build_score_hook(
    identity: Any, input_sha: str, summaries: Mapping[str, object], gates: Mapping[int, Mapping[str, object]],
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle score identity type drift")
    return _score_payload(identity, input_sha, summaries, gates)


def _validate_score_hook(
    value: Mapping[str, object], identity: Any, input_payload_value: Mapping[str, object], input_sha: str,
    fixed: Any,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity) or not isinstance(fixed, shared_score.FixedEvaluationAuthority):
        raise SpeedSmokeError("speed-smoke lifecycle score typed boundary drift")
    return validate_score_payload(
        value, identity=identity, input_payload_value=input_payload_value,
        input_authority_sha256=input_sha, evaluation_authority=fixed,
    )


def _terminal_verdict_hook(gates: Mapping[int, Mapping[str, object]]) -> str:
    return terminal_verdict(gates)


def _make_terminal_hook(
    identity: Any, attempt_sha: str, input_sha: str, score_sha: str, gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle terminal identity type drift")
    return _terminal_payload(identity, attempt_sha, input_sha, score_sha, gates, verdict)


def _validate_terminal_hook(
    value: Mapping[str, object], identity: Any, attempt_sha: str, input_sha: str,
    score_payload_value: Mapping[str, object], score_sha: str,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle terminal identity type drift")
    return validate_terminal_payload(
        value, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
        score_payload=score_payload_value, score_sha256=score_sha,
    )


def _make_failure_hook(
    identity: Any, attempt_sha: str, input_sha: str | None, stage: str, error: BaseException, progress: object | None,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle failure identity type drift")
    return _failure_payload(identity, attempt_sha, input_sha, stage, error, progress)


def _validate_failure_hook(
    value: Mapping[str, object], identity: Any, attempt_sha: str, input_sha: str | None,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke lifecycle failure identity type drift")
    return validate_failure_payload(value, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha)


def _validate_reserved_score_artifact(root: Path, artifact: shared_score.ArtifactRoot, identity: Any) -> None:
    if not isinstance(identity, SpeedSmokeIdentity):
        raise SpeedSmokeError("speed-smoke reserved artifact identity type drift")
    try:
        shared_score.validate_reserved_artifact_root(
            Path(root), artifact, root_relative=plan.SCORE_ROOT_RELATIVE, topology=SCORE_TOPOLOGY,
        )
    except shared_score.ScoreError as error:
        raise SpeedSmokeError(str(error)) from error


SPEED_SMOKE_LIFECYCLE_HOOKS = shared_score.ProfiledLifecycleHooks(
    route="precision_aware_causal_dual_memory_cell_d_speed_smoke_v1",
    score_root_relative=plan.SCORE_ROOT_RELATIVE,
    budgets=(4,),
    require_capability=shared_score.require_execution_capability,
    validate_preflight=_validate_preflight_hook,
    validate_authorization=_validate_authorization_hook,
    implementation_closure=_closure_hook,
    validate_source_gate=_source_gate_hook,
    assert_fresh_score_root=_fresh_score_hook,
    make_attempt=_attempt_payload,
    fixed_authority_from_preflight=_fixed_from_preflight,
    input_payload=_input_payload_hook,
    validate_input_payload=_validate_input_hook,
    summarize_budget=_summarize_hook,
    budget_gate=_budget_gate_hook,
    continue_after_budget=lambda _budget, _gates: True,
    build_score=_build_score_hook,
    validate_score=_validate_score_hook,
    terminal_verdict=_terminal_verdict_hook,
    make_terminal=_make_terminal_hook,
    validate_terminal=_validate_terminal_hook,
    make_failure=_make_failure_hook,
    validate_failure=_validate_failure_hook,
    validate_reserved_score_artifact=_validate_reserved_score_artifact,
)


def run_authorized_speed_smoke_lifecycle(
    root: Path, *, identity: SpeedSmokeIdentity, capability: object, backend: shared_score.ScoreBackend,
    artifact: shared_score.ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Future root-only launch adapter around the sole shared score lifecycle."""
    _validate_mutating_launch_environment(environ)
    _require_live_completed_v2(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    return shared_score.run_profiled_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=official_preflight_sha256, root_authorization_sha256=root_authorization_sha256,
        preflight=preflight, authorization=authorization, hooks=SPEED_SMOKE_LIFECYCLE_HOOKS,
    )
