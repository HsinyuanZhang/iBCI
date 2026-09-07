"""Production runtime for the frozen sub-M CO V9 three-arm matrix.

This is intentionally a small execution layer, not a replacement for the
synthetic V9 ledger.  It reuses the already-audited V3R2 data path:

* chronological first 30 rewarded trials build the B3 activity identity;
* the first 50 rewarded trials fit T4 / construct TS4;
* only windows strictly after rewarded trial 50 are decoded;
* T4, direct standardized zero4, and TS4 are evaluated with their own fixed
  epoch_011 shared-view checkpoints; and
* Phase A writes only prediction/target artifacts.  R2 is calculated only by
  :func:`finalize_artifacts` after all 270 cells are present.

No target-session optimizer, backward call, checkpoint selection, or online
state update is present here.  The module is deliberately usable on a staging
machine: every input is path- and SHA-bound in the immutable run manifest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Iterable, Mapping, Sequence


VIEWS = ("sua", "pseudo_mua")
ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
SEEDS = (42, 43, 44)
ACTIVITY_IDENTITY_TRIALS = 30
T4_FIT_POOL_TRIALS = 50
OUTPUT_DIM = 2
EXPECTED_SESSION_COUNT = 15
EXPECTED_CELL_COUNT = EXPECTED_SESSION_COUNT * len(VIEWS) * len(ARMS) * len(SEEDS)
RUNTIME_SCHEMA = "dandi_000688_subm_co_three_arm_v9_runtime_v1"
INPUT_MANIFEST_SCHEMA = "dandi_000688_subm_co_three_arm_v9_input_manifest_v1"
FROZEN_COHORT_SHA256 = "3ba5cb61601e53ae76db9d10317111cee9ff5c421ef360ed62d0a93d40906fec"
FROZEN_QUERY_MAP_SHA256 = "b45f988bb7eb393480a8244463ccd702fc5a99863c79b1256406f9da8c3a920c"
TORCHMETRICS_NEAR_CONSTANT_ATOL = 1.0e-4


class V9RuntimeError(RuntimeError):
    """A frozen V9 runtime invariant was broken."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V9RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_regular(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return path.is_file() and not path.is_symlink() and stat.S_IMODE(mode) == 0o444


def _write_exclusive(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Keep partial outputs as incidents.  A retry must never silently
        # replace an uncertain artifact.
        raise
    os.chmod(path, 0o444)
    _require(_readonly_regular(path), f"immutable write failed: {path}")
    return hashlib.sha256(raw).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_exclusive(path, canonical_bytes(dict(payload)))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V9RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


@dataclass(frozen=True, order=True)
class CohortSession:
    asset_id: str
    session_id: str
    frozen_path: str
    nwb_sha256: str
    nwb_bytes: int
    query_window_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class CellKey:
    asset_id: str
    session_id: str
    view: str
    arm: str
    seed: int

    def __post_init__(self) -> None:
        _require(self.view in VIEWS, f"invalid view: {self.view}")
        _require(self.arm in ARMS, f"invalid arm: {self.arm}")
        _require(self.seed in SEEDS, f"invalid seed: {self.seed}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def artifact_relative_path(self) -> str:
        return f"artifacts/{self.asset_id}/{self.view}/{self.arm}/seed_{self.seed}/predictions_targets.npz"

    @property
    def commit_relative_path(self) -> str:
        return f"commits/{self.asset_id}/{self.view}/{self.arm}/seed_{self.seed}.json"


@dataclass(frozen=True)
class CheckpointSpec:
    arm: str
    seed: int
    path: str
    sha256: str
    bytes: int
    closure_path: str
    closure_sha256: str
    closure_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeContract:
    cohort: tuple[CohortSession, ...]
    checkpoints: tuple[CheckpointSpec, ...]
    teacher_path: str
    teacher_sha256: str
    teacher_bytes: int
    behavior_normalizer_paths: Mapping[str, str]
    behavior_normalizer_sha256: Mapping[str, str]
    side_normalizer_paths: Mapping[str, str]
    side_normalizer_sha256: Mapping[str, str]
    cohort_receipt_sha256: str
    scope_manifest_sha256: str
    query_map_sha256: str

    def __post_init__(self) -> None:
        _require(len(self.cohort) == EXPECTED_SESSION_COUNT, "V9 requires frozen 15-session cohort")
        _require(len({row.asset_id for row in self.cohort}) == EXPECTED_SESSION_COUNT, "duplicate cohort asset")
        _require(len(self.checkpoints) == len(ARMS) * len(SEEDS), "V9 requires nine epoch_011 checkpoints")
        _require(
            {(row.arm, row.seed) for row in self.checkpoints}
            == {(arm, seed) for arm in ARMS for seed in SEEDS},
            "checkpoint arm/seed topology drift",
        )

    @property
    def expected_keys(self) -> tuple[CellKey, ...]:
        return tuple(
            CellKey(row.asset_id, row.session_id, view, arm, seed)
            for row in self.cohort
            for view in VIEWS
            for arm in ARMS
            for seed in SEEDS
        )

    def cohort_row(self, asset_id: str) -> CohortSession:
        matches = [row for row in self.cohort if row.asset_id == asset_id]
        _require(len(matches) == 1, f"unknown cohort asset: {asset_id}")
        return matches[0]

    def checkpoint(self, arm: str, seed: int) -> CheckpointSpec:
        matches = [row for row in self.checkpoints if row.arm == arm and row.seed == seed]
        _require(len(matches) == 1, f"missing checkpoint for {arm}/seed={seed}")
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_SCHEMA,
            "cohort": [row.as_dict() for row in self.cohort],
            "models": [row.as_dict() for row in self.checkpoints],
            "teacher": {
                "path": self.teacher_path,
                "sha256": self.teacher_sha256,
                "bytes": self.teacher_bytes,
            },
            "normalizers": {
                view: {
                    "behavior_path": self.behavior_normalizer_paths[view],
                    "behavior_sha256": self.behavior_normalizer_sha256[view],
                    "side_path": self.side_normalizer_paths[view],
                    "side_sha256": self.side_normalizer_sha256[view],
                }
                for view in VIEWS
            },
            "preflight": {
                "receipt_sha256": self.cohort_receipt_sha256,
                "scope_manifest_sha256": self.scope_manifest_sha256,
                "query_map_sha256": self.query_map_sha256,
            },
            "topology": {
                "views": list(VIEWS),
                "arms": list(ARMS),
                "seeds": list(SEEDS),
                "expected_cells": EXPECTED_CELL_COUNT,
            },
            "chronology": {
                "activity_identity_trials": ACTIVITY_IDENTITY_TRIALS,
                "t4_fit_pool_trials": T4_FIT_POOL_TRIALS,
                "query_rule": "strictly_after_rewarded_trial_50",
                "selection": "chronological_first",
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def _file_pin(path: Path) -> tuple[str, int]:
    _require(path.is_file() and not path.is_symlink(), f"missing or unsafe input: {path}")
    return sha256_file(path), int(path.stat().st_size)


def _normalizer_paths(root: Path, pins: Any, view: str) -> tuple[Path, Path]:
    behavior = (root / pins[view].behavior.relative_path).resolve()
    side = (root / pins[view].side_feature.relative_path).resolve()
    return behavior, side


def _validate_zero4_closure(*, checkpoint: Path, checkpoint_sha256: str, checkpoint_bytes: int, closure: Path, seed: int) -> None:
    """Validate ordinary closure manifests and the seed-44 recovery receipt.

    The recovery receipt has a different top-level schema, but both forms
    contain a nested seed-specific terminal-checkpoint binding.  Validate that
    binding directly instead of assuming a standard seed-44 closure exists.
    """
    _require(checkpoint.name == "epoch_011.ckpt", f"zero4 must bind terminal epoch_011: {checkpoint}")
    payload = _read_json(closure)
    nodes: list[Mapping[str, Any]] = []
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            nodes.append(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    statuses = [str(node.get("status")) for node in nodes if isinstance(node.get("status"), str)]
    _require(any(status.startswith("completed") for status in statuses), f"zero4 closure is not completed: {closure}")
    matched = False
    for node in nodes:
        if node.get("seed") != seed:
            continue
        terminal = node.get("terminal_checkpoint")
        if isinstance(terminal, Mapping) and terminal.get("sha256") == checkpoint_sha256 and terminal.get("size_bytes") == checkpoint_bytes:
            matched = True
            break
    _require(matched, f"zero4 closure does not bind seed-{seed} terminal checkpoint: {closure}")


def make_runtime_contract(
    *,
    repo_root: Path,
    nwb_root: Path,
    zero4_checkpoints: Mapping[int, Path],
    closure_metadata: Mapping[tuple[str, int], Path],
    teacher_checkpoint: Path | None = None,
) -> RuntimeContract:
    """Read-only preflight that locks all actual V9 inputs before a forward.

    The old V1 module is used only for its literal, score-blind frozen cohort
    and T4/TS4 file pins.  This runtime does not invoke V1 authorization or
    scoring code and intentionally does not use its obsolete two-arm output
    machinery.
    """
    repo_root = repo_root.resolve()
    nwb_root = nwb_root.resolve()
    import sys

    for candidate in (repo_root / "sua_exploration", repo_root / "sua_exploration/scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from sua_exploration.mc_maze import subm_co_score_only as v1

    receipt_path = repo_root / v1.AUTHORITY_PINS["schema_preflight_receipt_v2"].relative_path
    scope_path = repo_root / v1.AUTHORITY_PINS["scope_freeze_manifest_v2"].relative_path
    draft_path = repo_root / "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v3r2/prelaunch_authorization_draft.json"
    receipt = _read_json(receipt_path)
    scope = _read_json(scope_path)
    draft = _read_json(draft_path)
    frozen = v1._assert_literal_cohort(receipt, scope)
    try:
        score_only_v2 = draft["authority"]["score_only_v2"]
        draft_cohort = score_only_v2["frozen_cohort"]
        draft_query_map = score_only_v2["matrix"]["query_window_count_by_asset_id"]
    except (KeyError, TypeError) as exc:
        raise V9RuntimeError("frozen V3R2 cohort/query audit is malformed") from exc
    _require(canonical_sha256(draft_cohort) == FROZEN_COHORT_SHA256, "frozen 15-row cohort SHA drift")
    _require(canonical_sha256(draft_query_map) == FROZEN_QUERY_MAP_SHA256, "frozen query-map SHA drift")
    _require(isinstance(draft_cohort, list) and isinstance(draft_query_map, Mapping), "frozen cohort/query shape drift")
    ledger = {str(row["asset_id"]): row for row in receipt["asset_disposition_ledger"]}
    cohort: list[CohortSession] = []
    for row in frozen:
        ledger_row = ledger.get(row.asset_id)
        _require(isinstance(ledger_row, Mapping), f"missing preflight row: {row.asset_id}")
        query_count = draft_query_map.get(row.asset_id)
        _require(isinstance(query_count, int) and query_count > 0, f"invalid query count: {row.asset_id}")
        nwb_path = nwb_root / row.frozen_path
        observed_sha, observed_bytes = _file_pin(nwb_path)
        _require(observed_sha == row.nwb_sha256 and observed_bytes == row.nwb_bytes, f"NWB pin drift: {row.asset_id}")
        cohort.append(
            CohortSession(
                row.asset_id, row.session_id, row.frozen_path, row.nwb_sha256, row.nwb_bytes, query_count
            )
        )
    _require(sum(row.query_window_count for row in cohort) == 708795, "unexpected frozen total query count")
    _require(
        [
            {"asset_id": row.asset_id, "frozen_path": row.frozen_path, "nwb_bytes": row.nwb_bytes,
             "nwb_sha256": row.nwb_sha256, "session_id": row.session_id}
            for row in cohort
        ] == draft_cohort,
        "runtime cohort differs from frozen V3R2 cohort",
    )

    checkpoints: list[CheckpointSpec] = []
    for pin in v1.TERMINAL_CHECKPOINTS:
        source_path = (repo_root / pin.checkpoint.relative_path).resolve()
        observed_sha, observed_bytes = _file_pin(source_path)
        _require(
            observed_sha == pin.checkpoint.sha256 and observed_bytes == pin.checkpoint.bytes,
            f"checkpoint pin drift: {pin.arm}/seed={pin.seed}",
        )
        closure_path = Path(closure_metadata[(pin.arm, pin.seed)]).expanduser().resolve()
        closure_sha, closure_bytes = _file_pin(closure_path)
        checkpoints.append(
            CheckpointSpec(
                pin.arm, pin.seed, str(source_path), observed_sha, observed_bytes,
                str(closure_path), closure_sha, closure_bytes,
            )
        )
    for seed in SEEDS:
        path = Path(zero4_checkpoints[seed]).expanduser().resolve()
        observed_sha, observed_bytes = _file_pin(path)
        closure_path = Path(closure_metadata[("shared_zero4", seed)]).expanduser().resolve()
        closure_sha, closure_bytes = _file_pin(closure_path)
        _validate_zero4_closure(
            checkpoint=path,
            checkpoint_sha256=observed_sha,
            checkpoint_bytes=observed_bytes,
            closure=closure_path,
            seed=seed,
        )
        checkpoints.append(CheckpointSpec(
            "shared_zero4", seed, str(path), observed_sha, observed_bytes,
            str(closure_path), closure_sha, closure_bytes,
        ))

    teacher_pin = v1.TEACHER_CHECKPOINT
    teacher_path = Path(teacher_checkpoint or (repo_root / teacher_pin.relative_path)).expanduser().resolve()
    teacher_sha, teacher_bytes = _file_pin(teacher_path)
    _require(
        teacher_sha == teacher_pin.sha256 and teacher_bytes == teacher_pin.bytes,
        "teacher checkpoint pin drift",
    )
    behavior_paths: dict[str, str] = {}
    behavior_sha: dict[str, str] = {}
    side_paths: dict[str, str] = {}
    side_sha: dict[str, str] = {}
    for view in VIEWS:
        behavior, side = _normalizer_paths(repo_root, v1.NORMALIZER_PINS, view)
        observed_behavior_sha, observed_behavior_bytes = _file_pin(behavior)
        observed_side_sha, observed_side_bytes = _file_pin(side)
        _require(
            observed_behavior_sha == v1.NORMALIZER_PINS[view].behavior.sha256
            and observed_behavior_bytes == v1.NORMALIZER_PINS[view].behavior.bytes,
            f"behavior normalizer pin drift: {view}",
        )
        _require(
            observed_side_sha == v1.NORMALIZER_PINS[view].side_feature.sha256
            and observed_side_bytes == v1.NORMALIZER_PINS[view].side_feature.bytes,
            f"side normalizer pin drift: {view}",
        )
        behavior_paths[view], behavior_sha[view] = str(behavior), observed_behavior_sha
        side_paths[view], side_sha[view] = str(side), observed_side_sha
    return RuntimeContract(
        tuple(cohort), tuple(checkpoints), str(teacher_path), teacher_sha, teacher_bytes,
        behavior_paths, behavior_sha, side_paths, side_sha,
        sha256_file(receipt_path), sha256_file(scope_path), FROZEN_QUERY_MAP_SHA256,
    )


def verify_runtime_inputs(*, contract: RuntimeContract, nwb_root: Path) -> None:
    """Rehash every matrix input immediately before a real forward/finalize."""
    nwb_root = nwb_root.resolve()
    for row in contract.cohort:
        observed_sha, observed_bytes = _file_pin(nwb_root / row.frozen_path)
        _require(
            (observed_sha, observed_bytes) == (row.nwb_sha256, row.nwb_bytes),
            f"NWB changed after V9 input manifest: {row.asset_id}",
        )
    for spec in contract.checkpoints:
        observed_sha, observed_bytes = _file_pin(Path(spec.path))
        _require(
            (observed_sha, observed_bytes) == (spec.sha256, spec.bytes),
            f"checkpoint changed after V9 input manifest: {spec.arm}/{spec.seed}",
        )
        closure_sha, closure_bytes = _file_pin(Path(spec.closure_path))
        _require(
            (closure_sha, closure_bytes) == (spec.closure_sha256, spec.closure_bytes),
            f"closure changed after V9 input manifest: {spec.arm}/{spec.seed}",
        )
    teacher_sha, teacher_bytes = _file_pin(Path(contract.teacher_path))
    _require((teacher_sha, teacher_bytes) == (contract.teacher_sha256, contract.teacher_bytes), "teacher changed after V9 input manifest")
    for view in VIEWS:
        behavior_sha, _behavior_bytes = _file_pin(Path(contract.behavior_normalizer_paths[view]))
        side_sha, _side_bytes = _file_pin(Path(contract.side_normalizer_paths[view]))
        _require(behavior_sha == contract.behavior_normalizer_sha256[view], f"behavior normalizer changed: {view}")
        _require(side_sha == contract.side_normalizer_sha256[view], f"side normalizer changed: {view}")


def write_input_manifest(path: Path, contract: RuntimeContract) -> str:
    """Seal a portable, explicit nine-slot V9 input manifest once."""
    path = path.resolve()
    runtime_contract = contract.as_dict()
    payload = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "status": "FROZEN_INPUTS_READY_FOR_FORWARD_ONLY_V9",
        "runtime_contract": runtime_contract,
        "contract_sha256": contract.sha256,
        # These duplicated top-level fields make the actual source of every
        # scientific input inspectable without a custom contract decoder.
        "cohort": runtime_contract["cohort"],
        "models": runtime_contract["models"],
        "teacher": runtime_contract["teacher"],
        "normalizers": runtime_contract["normalizers"],
        "budgets": runtime_contract["chronology"],
        "metric": {
            "phase_a": "prediction_target_artifacts_only_no_metric",
            "phase_b": "validated_numpy_approximation_of_torchmetrics_1_5_1_variance_weighted_cpu_float32_atol_2e-6_pending_local_authoritative_recompute",
        },
        "preflight": runtime_contract["preflight"],
    }
    if path.exists():
        _require(_readonly_regular(path), "existing input manifest is unsafe")
        _require(_read_json(path) == payload, "existing input manifest differs from current pins")
        return sha256_file(path)
    digest = _write_json_exclusive(path, payload)
    _require(_read_json(path) == payload and sha256_file(path) == digest, "input manifest reopen/hash failure")
    return digest


def _runtime_contract_from_mapping(value: Mapping[str, Any]) -> RuntimeContract:
    try:
        normalizers = value["normalizers"]
        return RuntimeContract(
            cohort=tuple(CohortSession(**dict(row)) for row in value["cohort"]),
            checkpoints=tuple(CheckpointSpec(**dict(row)) for row in value["models"]),
            teacher_path=str(value["teacher"]["path"]),
            teacher_sha256=str(value["teacher"]["sha256"]),
            teacher_bytes=int(value["teacher"]["bytes"]),
            behavior_normalizer_paths={view: str(normalizers[view]["behavior_path"]) for view in VIEWS},
            behavior_normalizer_sha256={view: str(normalizers[view]["behavior_sha256"]) for view in VIEWS},
            side_normalizer_paths={view: str(normalizers[view]["side_path"]) for view in VIEWS},
            side_normalizer_sha256={view: str(normalizers[view]["side_sha256"]) for view in VIEWS},
            cohort_receipt_sha256=str(value["preflight"]["receipt_sha256"]),
            scope_manifest_sha256=str(value["preflight"]["scope_manifest_sha256"]),
            query_map_sha256=str(value["preflight"]["query_map_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V9RuntimeError("malformed V9 runtime contract in input manifest") from exc


def load_input_manifest(path: Path, *, require_frozen_cohort: bool = True) -> tuple[RuntimeContract, str]:
    """Open and cross-check a frozen input manifest before production work."""
    path = path.resolve()
    _require(_readonly_regular(path), f"input manifest must be immutable 0444: {path}")
    raw = path.read_bytes()
    payload = _read_json(path)
    _require(canonical_bytes(payload) == raw, "input manifest is not canonical JSON")
    _require(payload.get("schema") == INPUT_MANIFEST_SCHEMA, "input manifest schema drift")
    _require(payload.get("status") == "FROZEN_INPUTS_READY_FOR_FORWARD_ONLY_V9", "input manifest status drift")
    runtime_contract = payload.get("runtime_contract")
    _require(isinstance(runtime_contract, Mapping), "input manifest misses runtime contract")
    contract = _runtime_contract_from_mapping(runtime_contract)
    _require(contract.sha256 == payload.get("contract_sha256"), "input manifest contract SHA drift")
    _require(payload.get("cohort") == runtime_contract.get("cohort"), "input manifest cohort duplicate drift")
    _require(payload.get("models") == runtime_contract.get("models"), "input manifest model duplicate drift")
    _require(payload.get("teacher") == runtime_contract.get("teacher"), "input manifest teacher duplicate drift")
    _require(payload.get("normalizers") == runtime_contract.get("normalizers"), "input manifest normalizer duplicate drift")
    _require(payload.get("budgets") == runtime_contract.get("chronology"), "input manifest budget duplicate drift")
    cohort_payload = [
        {
            "asset_id": row.asset_id,
            "frozen_path": row.frozen_path,
            "nwb_bytes": row.nwb_bytes,
            "nwb_sha256": row.nwb_sha256,
            "session_id": row.session_id,
        }
        for row in contract.cohort
    ]
    query_map = {row.asset_id: row.query_window_count for row in contract.cohort}
    if require_frozen_cohort:
        _require(len(query_map) == EXPECTED_SESSION_COUNT and sum(query_map.values()) == 708795, "input manifest frozen query cardinality drift")
        _require(canonical_sha256(cohort_payload) == FROZEN_COHORT_SHA256, "input manifest cohort does not reconstruct frozen 15-row SHA")
        _require(canonical_sha256(query_map) == FROZEN_QUERY_MAP_SHA256, "input manifest query map does not reconstruct frozen SHA")
        _require(contract.query_map_sha256 == FROZEN_QUERY_MAP_SHA256, "input manifest query-map identity drift")
    else:
        _require(contract.query_map_sha256 == canonical_sha256(query_map), "synthetic input manifest query-map mismatch")
    return contract, hashlib.sha256(raw).hexdigest()


def _np() -> Any:
    import numpy as np
    return np


def _runtime_owners(repo_root: Path) -> dict[str, Any]:
    import sys
    for candidate in (
        repo_root / "sua_exploration",
        repo_root / "sua_exploration/scripts",
        repo_root / "streaming_calibration_exp",
    ):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3
    return parity_v3._runtime_owners()


def _load_mean_std(path: str, *, label: str) -> tuple[Any, Any]:
    np = _np()
    try:
        with np.load(path, allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float32)
            std = np.asarray(archive["std"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise V9RuntimeError(f"invalid {label} normalizer: {path}") from exc
    _require(mean.ndim == 1 and mean.shape == std.shape and bool(np.isfinite(mean).all()) and bool(np.all(std > 0)), f"invalid {label} normalizer values")
    return mean, std


def _build_base(
    *, repo_root: Path, nwb_path: Path, view: str, behavior_mean: Any, behavior_std: Any, owners: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    """Call the audited V3R2 owner route exactly once for one session/view."""
    from sua_exploration.mc_maze import subm_co_three_arm_score_only_v9 as v9
    from sua_exploration.mc_maze import subm_co_score_only_v3r2 as v3r2

    record, rebuilt, _builder_trials, bridge_trace = v3r2._build_view_base(
        nwb_path=nwb_path,
        view=view,
        normalizer={"behavior_mean": behavior_mean, "behavior_std": behavior_std},
        owners=owners,
    )
    _require(int(rebuilt.shape[0]) == ACTIVITY_IDENTITY_TRIALS, "activity identity is not chronological first 30")
    _require(int(record.valid_starts.size) > 0, "post-50 query is empty")
    _require(int(record.valid_starts.size) == int(bridge_trace["query_window_count"]), "base/query bridge count drift")
    # The v3r2 route verifies this internally.  Keep the direct declaration in
    # the artifact evidence so future readers do not have to infer the timeline.
    evidence = {
        "activity_identity_trials": ACTIVITY_IDENTITY_TRIALS,
        "t4_fit_pool_trials": T4_FIT_POOL_TRIALS,
        "query_rule": "strictly_after_rewarded_trial_50",
        "bridge": bridge_trace,
        "synthetic_v9_core_available": v9.EXPECTED_CELL_COUNT == EXPECTED_CELL_COUNT,
    }
    return record, rebuilt, evidence


def _dataset_for_descriptor(
    *,
    nwb_path: Path,
    view: str,
    arm: str,
    seed: int,
    record: Any,
    rebuilt_calibration: Any,
    side_mean: Any | None,
    side_std: Any | None,
    owners: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Construct one evaluation dataset without changing the online activity path."""
    np = _np()
    from sua_exploration.mc_maze import subm_co_score_only_v3r2 as v3r2
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3

    if arm in {"shared_t4", "shared_ts4"}:
        _require(side_mean is not None and side_std is not None, "T4/TS4 requires source side normalizer")
        dataset, n_channels = v3r2._dataset_for_cell(
            nwb_path=nwb_path,
            view=view,
            arm=arm,
            seed=seed,
            record=record,
            rebuilt_calibration=rebuilt_calibration,
            normalizer={"side_mean": side_mean, "side_std": side_std},
            owners=owners,
        )
        return dataset, {
            "descriptor": "t4" if arm == "shared_t4" else "ts4",
            "channel_count": n_channels,
            "target_direction_label_reads_for_descriptor": "owned_by_t4_loader",
            "source_side_normalizer_used": True,
        }
    _require(arm == "shared_zero4", f"unknown V9 arm: {arm}")
    # Deliberately create the direct standardized neutral point from N only.
    # This branch never receives side_mean, side_std, a T4 loader, labels, or
    # per-trial rates.
    n_channels = int(record.neural.shape[1])
    _require(0 < n_channels < 100, "zero4 channel count must satisfy 0 < N < 100")
    zero4 = np.zeros((n_channels, 4), dtype=np.float32)
    _require(bool(np.all(zero4.view(np.uint32) == np.uint32(0))), "zero4 lost positive float32-zero bits")
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=rebuilt_calibration,
        window_size=parity_v3.HISTORY_BINS,
        session_name=record.name,
        side_features=zero4,
        electrode_ids=None,
    )
    _require(len(dataset) == int(record.valid_starts.size), "zero4 dataset/query count drift")
    return dataset, {
        "descriptor": "direct_standardized_zero4",
        "channel_count": n_channels,
        "target_direction_label_reads_for_descriptor": 0,
        "t4_trial_rate_reads_for_descriptor": 0,
        "target_t4_rate_fit_calls": 0,
        "source_side_normalizer_used": False,
        "bitwise_positive_float32_zero": True,
    }


def _unpack_batch(batch: Sequence[Any]) -> tuple[Any, Any, Any, Any, Any]:
    if len(batch) == 6:
        neural, behavior, calibration, _names, side, electrodes = batch
        return neural, behavior, calibration, side, electrodes
    if len(batch) == 5:
        neural, behavior, calibration, _names, side = batch
        return neural, behavior, calibration, side, None
    raise V9RuntimeError(f"unexpected B3S batch arity: {len(batch)}")


def collect_forward_predictions(*, model: Any, dataset: Any, device: str, owners: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """The exact score-only forward collector, with all metric calls removed."""
    np = _np()
    torch = owners["torch"]
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA device requested but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.synchronize(runtime_device)
    model.eval()
    for parameter in model.parameters():
        _require(parameter.requires_grad is False, "target evaluator exposes trainable model parameter")
    loader = owners["DataLoader"](dataset, batch_size=128, shuffle=False, num_workers=0)
    predictions: list[Any] = []
    targets: list[Any] = []
    t0 = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            neural, behavior, calibration, side, electrodes = _unpack_batch(batch)
            neural = neural.to(runtime_device)
            behavior = behavior.to(runtime_device)
            calibration = calibration.to(runtime_device)
            side = side.to(runtime_device)
            electrodes = None if electrodes is None else electrodes.to(runtime_device)
            decoder_key_features = model.decoder_key_features(side)
            raw_prediction, _ = model.student(
                neural,
                calib_trials=calibration,
                side_features=side,
                decoder_key_features=decoder_key_features,
                electrode_ids=electrodes,
            )
            prediction = raw_prediction[:, -1:, :] / 5.0
            target = behavior[:, -1:, :]
            predictions.append(prediction[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False))
            targets.append(target[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False))
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    prediction_np = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    target_np = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    _require(prediction_np.shape == target_np.shape == (len(dataset), OUTPUT_DIM), "forward output shape/count drift")
    _require(bool(np.isfinite(prediction_np).all()) and bool(np.isfinite(target_np).all()), "nonfinite forward output")
    return prediction_np, target_np, {
        "forward_seconds": time.monotonic() - t0,
        "device": str(runtime_device),
        "batches": int(math.ceil(len(dataset) / 128)),
        "metric_computed": False,
        "backward_called": False,
    }


def _write_npz_exclusive(path: Path, predictions: Any, targets: Any) -> str:
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, predictions=predictions, targets=targets)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    os.chmod(path, 0o444)
    _require(_readonly_regular(path), f"immutable artifact write failed: {path}")
    return sha256_file(path)


def _safe_output_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise V9RuntimeError(f"artifact path escapes output root: {relative}") from exc
    return candidate


def initialize_output(
    output_root: Path,
    contract: RuntimeContract,
    *,
    input_manifest_path: Path,
    input_manifest_sha256: str,
    execution_device: Mapping[str, Any],
) -> None:
    """Create one immutable Phase-A manifest, or verify an exact resume."""
    output_root = output_root.resolve()
    manifest = output_root / "run_manifest.json"
    payload = {
        "schema": RUNTIME_SCHEMA,
        "status": "PHASE_A_FORWARD_ONLY_NO_METRIC",
        "contract": contract.as_dict(),
        "contract_sha256": contract.sha256,
        "input_manifest_path": str(input_manifest_path.resolve()),
        "input_manifest_sha256": input_manifest_sha256,
        "execution_device": dict(execution_device),
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "phase_a_metric_policy": "FORBIDDEN_UNTIL_ALL_270_ARTIFACTS_COMMITTED",
        "overwrite_policy": "immutable_missing_only",
    }
    if manifest.exists():
        _require(_readonly_regular(manifest), "unsafe existing V9 manifest")
        existing = _read_json(manifest)
        _require(existing == payload, "existing V9 output contract drift")
        return
    _write_json_exclusive(manifest, payload)


def execution_device_binding(device: str) -> dict[str, Any]:
    """A small immutable runtime binding that prevents mixed-device resumes."""
    import torch

    runtime_device = torch.device(device)
    binding: dict[str, Any] = {
        "device": str(runtime_device),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "tf32_disabled": True,
        "deterministic_algorithms": True,
        "batch_size": 128,
    }
    if runtime_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA execution binding requested but CUDA is unavailable")
        binding["cuda_device_name"] = str(torch.cuda.get_device_name(runtime_device))
    return binding


def _read_artifact(path: Path, *, expected_rows: int | None = None) -> tuple[Any, Any]:
    np = _np()
    _require(_readonly_regular(path), f"missing or unsafe artifact: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive["predictions"]
            target = archive["targets"]
    except (OSError, ValueError, KeyError) as exc:
        raise V9RuntimeError(f"cannot read artifact {path}") from exc
    _require(isinstance(prediction, np.ndarray) and isinstance(target, np.ndarray), "artifact arrays missing")
    _require(prediction.dtype == np.dtype(np.float32) and target.dtype == np.dtype(np.float32), "artifact dtype must be exact float32")
    _require(prediction.flags.c_contiguous and target.flags.c_contiguous, "artifact arrays must be C-contiguous")
    _require(prediction.ndim == target.ndim == 2 and prediction.shape == target.shape and prediction.shape[1] == OUTPUT_DIM, "artifact shape drift")
    if expected_rows is not None:
        _require(prediction.shape[0] == expected_rows, "artifact query-window count drift")
    _require(bool(np.isfinite(prediction).all()) and bool(np.isfinite(target).all()), "artifact contains nonfinite values")
    return np.ascontiguousarray(prediction), np.ascontiguousarray(target)


def _validate_committed_cell(*, output_root: Path, contract: RuntimeContract, key: CellKey) -> tuple[Any, Any]:
    """Reopen one existing cell before resume or final aggregation."""
    row = contract.cohort_row(key.asset_id)
    artifact_path = _safe_output_path(output_root, key.artifact_relative_path)
    commit_path = _safe_output_path(output_root, key.commit_relative_path)
    _require(_readonly_regular(artifact_path) and _readonly_regular(commit_path), f"orphan/unsafe existing V9 cell: {key}")
    raw = commit_path.read_bytes()
    commit = _read_json(commit_path)
    _require(canonical_bytes(commit) == raw, "existing commit is not canonical JSON")
    _require(
        commit.get("schema") == RUNTIME_SCHEMA
        and commit.get("status") == "ARTIFACT_COMMITTED_NO_METRIC"
        and commit.get("contract_sha256") == contract.sha256
        and commit.get("cell") == key.as_dict()
        and commit.get("query_window_count") == row.query_window_count
        and commit.get("metric_computed") is False,
        "existing V9 commit identity/policy drift",
    )
    artifact = commit.get("artifact")
    _require(isinstance(artifact, Mapping), "existing V9 artifact pin missing")
    _require(
        artifact.get("relative_path") == key.artifact_relative_path
        and artifact.get("bytes") == artifact_path.stat().st_size
        and artifact.get("dtype") == "float32"
        and artifact.get("shape") == [row.query_window_count, OUTPUT_DIM]
        and artifact.get("sha256") == sha256_file(artifact_path),
        "existing V9 artifact pin drift",
    )
    prediction, target = _read_artifact(artifact_path, expected_rows=row.query_window_count)
    _require(
        commit.get("query_behavior_trace_sha256") == hashlib.sha256(target.tobytes(order="C")).hexdigest(),
        "existing V9 target trace drift",
    )
    return prediction, target


def write_cell_artifact(
    *, output_root: Path, contract: RuntimeContract, key: CellKey, predictions: Any, targets: Any, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal one Phase-A result; the payload deliberately contains no metric."""
    np = _np()
    row = contract.cohort_row(key.asset_id)
    _require(key.session_id == row.session_id, "cell session identity drift")
    predictions = np.ascontiguousarray(predictions, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)
    _require(predictions.shape == targets.shape == (row.query_window_count, OUTPUT_DIM), "artifact query count drift")
    artifact = _safe_output_path(output_root, key.artifact_relative_path)
    commit = _safe_output_path(output_root, key.commit_relative_path)
    _require(not artifact.exists() and not commit.exists(), f"existing/orphan cell blocks overwrite: {key}")
    artifact_sha = _write_npz_exclusive(artifact, predictions, targets)
    reopened_prediction, reopened_target = _read_artifact(artifact, expected_rows=row.query_window_count)
    _require(
        reopened_prediction.tobytes(order="C") == predictions.tobytes(order="C")
        and reopened_target.tobytes(order="C") == targets.tobytes(order="C"),
        "artifact reopen differs from just-written arrays",
    )
    target_trace = hashlib.sha256(targets.tobytes(order="C")).hexdigest()
    payload = {
        "schema": RUNTIME_SCHEMA,
        "status": "ARTIFACT_COMMITTED_NO_METRIC",
        "contract_sha256": contract.sha256,
        "cell": key.as_dict(),
        "query_window_count": row.query_window_count,
        "artifact": {
            "relative_path": key.artifact_relative_path,
            "sha256": artifact_sha,
            "bytes": artifact.stat().st_size,
            "dtype": "float32",
            "shape": [row.query_window_count, OUTPUT_DIM],
        },
        "query_behavior_trace_sha256": target_trace,
        "evidence": dict(evidence),
        "metric_computed": False,
    }
    _write_json_exclusive(commit, payload)
    return {"artifact": str(artifact), "commit": str(commit), "query_window_count": row.query_window_count}


def _load_model(spec: CheckpointSpec, contract: RuntimeContract, device: str, owners: Mapping[str, Any]) -> Any:
    observed_sha, observed_bytes = _file_pin(Path(spec.path))
    _require(observed_sha == spec.sha256 and observed_bytes == spec.bytes, f"checkpoint changed before load: {spec.arm}/{spec.seed}")
    torch = owners["torch"]
    model_loader = owners["model"].load_frozen_model
    return model_loader(Path(spec.path), Path(contract.teacher_path), "B3S", torch.device(device))


def _base_trace(record: Any, rebuilt: Any) -> dict[str, Any]:
    np = _np()
    return {
        "neural": hashlib.sha256(np.ascontiguousarray(record.neural).tobytes()).hexdigest(),
        "behavior": hashlib.sha256(np.ascontiguousarray(record.behavior).tobytes()).hexdigest(),
        "valid_starts": hashlib.sha256(np.ascontiguousarray(record.valid_starts).tobytes()).hexdigest(),
        "activity_calibration": hashlib.sha256(np.ascontiguousarray(rebuilt).tobytes()).hexdigest(),
    }


def run_forward_cells(
    *,
    repo_root: Path,
    nwb_root: Path,
    output_root: Path,
    contract: RuntimeContract,
    device: str,
    input_manifest_path: Path,
    input_manifest_sha256: str,
    requested_keys: Iterable[CellKey] | None = None,
) -> list[dict[str, Any]]:
    """Run missing requested V9 Phase-A cells and write immutable artifacts."""
    import torch

    repo_root, nwb_root, output_root = repo_root.resolve(), nwb_root.resolve(), output_root.resolve()
    manifest_contract, actual_manifest_sha = load_input_manifest(input_manifest_path)
    _require(manifest_contract == contract, "caller contract differs from reopened V9 input manifest")
    _require(actual_manifest_sha == input_manifest_sha256, "caller input-manifest SHA differs from reopened manifest")
    verify_runtime_inputs(contract=contract, nwb_root=nwb_root)
    device_binding = execution_device_binding(device)
    initialize_output(
        output_root, contract,
        input_manifest_path=input_manifest_path, input_manifest_sha256=input_manifest_sha256,
        execution_device=device_binding,
    )
    expected = set(contract.expected_keys)
    keys = tuple(requested_keys if requested_keys is not None else contract.expected_keys)
    _require(keys and set(keys) <= expected, "requested cells fall outside frozen V9 matrix")
    owners = _runtime_owners(repo_root)
    behavior_stats = {view: _load_mean_std(contract.behavior_normalizer_paths[view], label=f"{view} behavior") for view in VIEWS}
    side_stats: dict[str, tuple[Any, Any]] = {}
    models: dict[tuple[str, int], Any] = {}
    completed: list[dict[str, Any]] = []
    # Session/view outer loops ensure the online neural activity and exactly
    # matched query targets are materialized once before three descriptor arms.
    requested_by_session_view: dict[tuple[str, str], list[CellKey]] = {}
    for key in keys:
        artifact_exists = _safe_output_path(output_root, key.artifact_relative_path).exists()
        commit_exists = _safe_output_path(output_root, key.commit_relative_path).exists()
        if artifact_exists or commit_exists:
            _require(artifact_exists and commit_exists, f"orphan existing V9 cell: {key}")
            _validate_committed_cell(output_root=output_root, contract=contract, key=key)
            continue
        requested_by_session_view.setdefault((key.asset_id, key.view), []).append(key)
    with torch.no_grad():
        for cohort_row in contract.cohort:
            nwb_path = nwb_root / cohort_row.frozen_path
            for view in VIEWS:
                keys_here = sorted(requested_by_session_view.get((cohort_row.asset_id, view), []))
                if not keys_here:
                    continue
                behavior_mean, behavior_std = behavior_stats[view]
                record, rebuilt, base_evidence = _build_base(
                    repo_root=repo_root, nwb_path=nwb_path, view=view,
                    behavior_mean=behavior_mean, behavior_std=behavior_std, owners=owners,
                )
                _require(record.name == cohort_row.session_id, f"runtime session identity drift: {record.name}")
                _require(int(record.valid_starts.size) == cohort_row.query_window_count, f"runtime query-map drift: {record.name}/{view}")
                trace = _base_trace(record, rebuilt)
                datasets: dict[tuple[str, int], tuple[Any, dict[str, Any]]] = {}
                for arm in sorted({key.arm for key in keys_here}):
                    if arm != "shared_zero4" and view not in side_stats:
                        side_stats[view] = _load_mean_std(contract.side_normalizer_paths[view], label=f"{view} T4")
                    descriptor_seeds = sorted({key.seed for key in keys_here if key.arm == arm})
                    if arm == "shared_t4":
                        # T4 has no evaluator-seed permutation.  V3R2 still
                        # validates its argument against the declared seed set,
                        # so build it once using the canonical representative
                        # seed 42 and reuse it for all three model seeds.
                        descriptor_seeds = [42]
                    elif arm == "shared_zero4":
                        descriptor_seeds = [0]
                    for seed in descriptor_seeds:
                        mean, std = side_stats[view] if arm != "shared_zero4" else (None, None)
                        datasets[(arm, seed)] = _dataset_for_descriptor(
                            nwb_path=nwb_path, view=view, arm=arm, seed=seed,
                            record=record, rebuilt_calibration=rebuilt,
                            side_mean=mean, side_std=std, owners=owners,
                        )
                for key in keys_here:
                    model_key = (key.arm, key.seed)
                    if model_key not in models:
                        models[model_key] = _load_model(contract.checkpoint(key.arm, key.seed), contract, device, owners)
                    descriptor_seed = key.seed if key.arm == "shared_ts4" else (42 if key.arm == "shared_t4" else 0)
                    descriptor_key = (key.arm, descriptor_seed)
                    dataset, descriptor_evidence = datasets[descriptor_key]
                    prediction, target, forward_evidence = collect_forward_predictions(
                        model=models[model_key], dataset=dataset, device=device, owners=owners
                    )
                    evidence = {
                        "checkpoint_sha256": contract.checkpoint(key.arm, key.seed).sha256,
                        "checkpoint_path": contract.checkpoint(key.arm, key.seed).path,
                        "base_input_trace": trace,
                        "base_protocol": base_evidence,
                        "descriptor": descriptor_evidence,
                        "forward": forward_evidence,
                    }
                    completed.append(write_cell_artifact(
                        output_root=output_root, contract=contract, key=key,
                        predictions=prediction, targets=target, evidence=evidence,
                    ))
    return completed


def _check_manifest(output_root: Path, contract: RuntimeContract) -> None:
    manifest = output_root / "run_manifest.json"
    _require(_readonly_regular(manifest), "missing immutable V9 manifest")
    payload = _read_json(manifest)
    _require(payload.get("contract_sha256") == contract.sha256, "V9 finalizer contract drift")
    input_path = Path(str(payload.get("input_manifest_path", ""))).expanduser().resolve()
    _require(_readonly_regular(input_path), "V9 input manifest is missing/unsafe during finalization")
    _require(sha256_file(input_path) == payload.get("input_manifest_sha256"), "V9 input manifest SHA drift during finalization")
    input_contract, _input_sha = load_input_manifest(input_path)
    _require(input_contract == contract, "V9 input manifest contract drift during finalization")


def validated_numpy_torchmetrics_151_approximation(prediction: Any, target: Any) -> float:
    """Validated NumPy approximation to TorchMetrics 1.5.1 R2Score.

    The remote stage has a newer TorchMetrics version.  This routine preserves
    the 1.5.1 float32 reduction and near-constant branches, and was validated
    against the local 1.5.1 implementation within absolute 2e-6.  It is an
    audit-only approximation; the final scientific endpoint must be reopened
    with local TorchMetrics 1.5.1 from the sealed artifacts.
    """
    np = _np()
    _require(
        isinstance(prediction, np.ndarray) and isinstance(target, np.ndarray)
        and prediction.dtype == target.dtype == np.dtype(np.float32)
        and prediction.flags.c_contiguous and target.flags.c_contiguous
        and prediction.ndim == target.ndim == 2 and prediction.shape == target.shape
        and prediction.shape[0] >= 2 and prediction.shape[1] == OUTPUT_DIM,
        "invalid float32 arrays for frozen TorchMetrics R2",
    )
    _require(bool(np.isfinite(prediction).all()) and bool(np.isfinite(target).all()), "nonfinite frozen TorchMetrics R2 arrays")
    count = np.float32(prediction.shape[0])
    sum_obs = np.sum(target, axis=0, dtype=np.float32)
    sum_squared_obs = np.sum(target * target, axis=0, dtype=np.float32)
    rss = np.sum((target - prediction) * (target - prediction), axis=0, dtype=np.float32)
    tss = sum_squared_obs - sum_obs * (sum_obs / count)
    cond_rss = np.logical_not(np.isclose(rss, np.zeros_like(rss), rtol=1.0e-5, atol=TORCHMETRICS_NEAR_CONSTANT_ATOL))
    cond_tss = np.logical_not(np.isclose(tss, np.zeros_like(tss), rtol=1.0e-5, atol=TORCHMETRICS_NEAR_CONSTANT_ATOL))
    raw_scores = np.ones_like(rss, dtype=np.float32)
    ordinary = np.logical_and(cond_rss, cond_tss)
    raw_scores[ordinary] = np.float32(1.0) - rss[ordinary] / tss[ordinary]
    raw_scores[np.logical_and(cond_rss, np.logical_not(cond_tss))] = np.float32(0.0)
    tss_sum = np.sum(tss, dtype=np.float32)
    _require(bool(np.isfinite(tss_sum)) and not bool(np.isclose(tss_sum, np.float32(0.0), rtol=1.0e-5, atol=TORCHMETRICS_NEAR_CONSTANT_ATOL)), "undefined frozen TorchMetrics R2 target variance")
    value = float(np.sum(tss / tss_sum * raw_scores, dtype=np.float32))
    _require(math.isfinite(value), "nonfinite frozen TorchMetrics R2")
    return value


def _observed_output_files(root: Path) -> set[str]:
    observed: set[str] = set()
    if not root.exists():
        return observed
    for path in root.rglob("*"):
        if path.is_symlink():
            raise V9RuntimeError(f"symlink forbidden in immutable output: {path}")
        if path.is_file():
            observed.add(str(path.relative_to(root)))
    return observed


def finalize_artifacts(*, output_root: Path, contract: RuntimeContract) -> dict[str, Any]:
    """CPU-only full-matrix R2 finalizer.  It refuses any partial matrix."""
    output_root = output_root.resolve()
    _check_manifest(output_root, contract)
    final_path = output_root / "aggregate" / "endpoint_aggregate.json"
    _require(not final_path.exists(), "final aggregate already exists; overwrite forbidden")
    allowed = {"run_manifest.json"}
    for key in contract.expected_keys:
        allowed.add(key.artifact_relative_path)
        allowed.add(key.commit_relative_path)
    _require(_observed_output_files(output_root) <= allowed, "unknown file in V9 artifact topology")
    rows: list[dict[str, Any]] = []
    target_by_asset: dict[str, bytes] = {}
    missing: list[CellKey] = []
    for key in contract.expected_keys:
        artifact_path = _safe_output_path(output_root, key.artifact_relative_path)
        commit_path = _safe_output_path(output_root, key.commit_relative_path)
        if not artifact_path.exists() or not commit_path.exists():
            missing.append(key)
            continue
        prediction, target = _validate_committed_cell(output_root=output_root, contract=contract, key=key)
        _require(int(target.shape[0]) == contract.cohort_row(key.asset_id).query_window_count, "finalizer query count drift")
        target_bytes = target.tobytes(order="C")
        previous = target_by_asset.setdefault(key.asset_id, target_bytes)
        _require(previous == target_bytes, f"cross-view/arm/seed target mismatch: {key.asset_id}")
        rows.append({
            "cell": key.as_dict(),
            "r2": validated_numpy_torchmetrics_151_approximation(prediction, target),
            "query_window_count": int(target.shape[0]),
        })
    _require(not missing and len(rows) == EXPECTED_CELL_COUNT, f"cannot finalize incomplete V9 matrix: {len(rows)}/{EXPECTED_CELL_COUNT}")
    by_arm_view_seed: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        by_arm_view_seed[arm] = {}
        for view in VIEWS:
            by_arm_view_seed[arm][view] = {}
            for seed in SEEDS:
                values = [
                    row["r2"] for row in rows
                    if row["cell"]["arm"] == arm and row["cell"]["view"] == view and row["cell"]["seed"] == seed
                ]
                _require(len(values) == EXPECTED_SESSION_COUNT, "aggregate cell cardinality drift")
                by_arm_view_seed[arm][view][str(seed)] = float(sum(values) / len(values))
    paired_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for label, left, right in (
        ("shared_t4_minus_shared_zero4", "shared_t4", "shared_zero4"),
        ("shared_t4_minus_shared_ts4", "shared_t4", "shared_ts4"),
    ):
        paired_contrasts[label] = {}
        for view in VIEWS:
            paired_contrasts[label][view] = {}
            for seed in SEEDS:
                left_by_asset = {
                    row["cell"]["asset_id"]: row["r2"] for row in rows
                    if row["cell"]["arm"] == left and row["cell"]["view"] == view and row["cell"]["seed"] == seed
                }
                right_by_asset = {
                    row["cell"]["asset_id"]: row["r2"] for row in rows
                    if row["cell"]["arm"] == right and row["cell"]["view"] == view and row["cell"]["seed"] == seed
                }
                _require(set(left_by_asset) == set(right_by_asset) == {row.asset_id for row in contract.cohort}, "paired contrast cohort drift")
                delta_by_asset = {asset: float(left_by_asset[asset] - right_by_asset[asset]) for asset in sorted(left_by_asset)}
                values = list(delta_by_asset.values())
                paired_contrasts[label][view][str(seed)] = {
                    "mean_delta_r2": float(sum(values) / len(values)),
                    "positive_session_count": sum(value > 0.0 for value in values),
                    "per_session_delta_r2": delta_by_asset,
                }
    payload = {
        "schema": RUNTIME_SCHEMA,
        "status": "FULL_270_CPU_APPROXIMATE_FINALIZED_PENDING_LOCAL_TORCHMETRICS_1_5_1",
        "contract_sha256": contract.sha256,
        "verified_cell_count": len(rows),
        "r2_definition": "validated_numpy_approximation_of_torchmetrics_1_5_1_variance_weighted_cpu_float32",
        "r2_validation_absolute_tolerance": 2.0e-6,
        "local_authoritative_torchmetrics_1_5_1_recompute_pending": True,
        "aggregate_recomputed_only_after_full_matrix": True,
        "mean_r2_by_arm_view_seed": by_arm_view_seed,
        "paired_contrasts": paired_contrasts,
        "cells": rows,
    }
    aggregate_sha = _write_json_exclusive(final_path, payload)
    return {"aggregate_path": str(final_path), "aggregate_sha256": aggregate_sha, "verified_cell_count": len(rows)}


def compare_cpu_gpu(
    *, model: Any, dataset: Any, owners: Mapping[str, Any], atol: float = 1.0e-5, rtol: float = 1.0e-5
) -> dict[str, Any]:
    """Forward-only parity using the exact same collector on both devices."""
    np = _np()
    torch = owners["torch"]
    _require(torch.cuda.is_available(), "GPU parity requested but CUDA unavailable")
    model.to(torch.device("cpu"))
    cpu_prediction, cpu_target, _ = collect_forward_predictions(model=model, dataset=dataset, device="cpu", owners=owners)
    cpu_r2 = validated_numpy_torchmetrics_151_approximation(cpu_prediction, cpu_target)
    model.to(torch.device("cuda:0"))
    gpu_prediction, gpu_target, _ = collect_forward_predictions(model=model, dataset=dataset, device="cuda:0", owners=owners)
    gpu_r2 = validated_numpy_torchmetrics_151_approximation(gpu_prediction, gpu_target)
    _require(cpu_target.tobytes(order="C") == gpu_target.tobytes(order="C"), "CPU/GPU target tensor mismatch")
    max_abs = float(np.max(np.abs(cpu_prediction.astype(np.float64) - gpu_prediction.astype(np.float64))))
    _require(bool(np.allclose(cpu_prediction, gpu_prediction, atol=atol, rtol=rtol)), f"CPU/GPU forward mismatch max_abs={max_abs}")
    return {
        "target_bitwise_equal": True,
        "prediction_max_abs": max_abs,
        "cpu_r2": cpu_r2,
        "gpu_r2": gpu_r2,
        "r2_abs_delta": abs(cpu_r2 - gpu_r2),
        "atol": atol,
        "rtol": rtol,
        "tf32_disabled": True,
        "deterministic_algorithms": True,
    }
