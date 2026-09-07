"""Synthetic-only V9 artifact ledger and scientific-matrix core.

This module deliberately stops before any external-data execution capability.
It has no NWB owner, checkpoint loader, model forward path, GPU path, formal
authorization, or formal R² result.  Its purpose is to make the scientific
V9 matrix mechanically testable before the nine terminal closures and the
Native-M2 release condition exist:

* exact 15 × 2 × 3 × 3 = 270 cell topology;
* fixed first-30 / first-50 / post-50 chronology declaration;
* artifact-only prediction/target commits with no partial metric;
* bitwise-paired ordered behavior targets across every view/arm/seed of one
  asset/session;
* strict no-overwrite and missing-only synthetic continuation;
* CPU-only synthetic finalization after all 270 commits; and
* direct, label-isolated positive-float32-zero4 construction.

``V9SyntheticRunToken`` is intentionally an *ordering token*, not a formal
capability.  It exists so unit tests can prove the authorization-first import
boundary without accidentally creating a caller-supplied trust-root scheme.
Any real detached authorization, source/closure verification, data adapter,
checkpoint loading, GPU parity, and external sub-M execution must be supplied
in a separately reviewed successor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


VIEWS = ("sua", "pseudo_mua")
ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
SEEDS = (42, 43, 44)
EXPECTED_SESSION_COUNT = 15
EXPECTED_CELL_COUNT = EXPECTED_SESSION_COUNT * len(VIEWS) * len(ARMS) * len(SEEDS)
ACTIVITY_IDENTITY_TRIALS = 30
T4_FIT_POOL_TRIALS = 50
QUERY_RULE = "strictly_after_rewarded_trial_50"
OUTPUT_DIM = 2
DEFAULT_SYNTHETIC_BOOTSTRAP_REPLICATES = 100_000
SYNTHETIC_SCOPE = "SYNTHETIC_TEST_ONLY_NOT_FORMAL_NOT_EXTERNAL"
_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+\Z")


class V9Error(RuntimeError):
    """Base V9 pre-execution contract error."""


class V9AuthorizationOrderError(V9Error):
    """The synthetic ordering guard was invoked after score runtime import."""


class V9ArtifactError(V9Error):
    """An immutable artifact or topology invariant failed."""


class V9IncompleteMatrixError(V9ArtifactError):
    """A metric/finalization action was attempted before all 270 commits."""


class V9ContinuationError(V9ArtifactError):
    """A synthetic continuation attempted to alter prior committed state."""


class V9Zero4Error(V9Error):
    """The direct zero4 descriptor contract was violated."""


def require(condition: bool, message: str, *, error: type[Exception] = V9Error) -> None:
    if not condition:
        raise error(message)


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used for every synthetic manifest/commit hash."""

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


def _safe_component(value: str, label: str) -> str:
    require(isinstance(value, str) and _COMPONENT_PATTERN.fullmatch(value) is not None, f"unsafe {label}", error=V9ArtifactError)
    return value


def _safe_relative(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute() and ".." not in candidate.parts, f"unsafe {label} path", error=V9ArtifactError)
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise V9ArtifactError(f"{label} escapes output root") from exc
    return resolved


def _is_readonly_regular(path: Path) -> bool:
    try:
        metadata = path.stat()
    except OSError:
        return False
    return path.is_file() and not path.is_symlink() and stat.S_IMODE(metadata.st_mode) == 0o444


def _write_exclusive(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial/non-read-only file is deliberately retained.  It becomes an
        # orphan incident rather than something a later continuation can erase.
        raise
    os.chmod(path, 0o444)
    require(_is_readonly_regular(path), f"immutable write failed: {path.name}", error=V9ArtifactError)
    return hashlib.sha256(raw).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_exclusive(path, canonical_bytes(dict(payload)))


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    require(_is_readonly_regular(path), f"missing/unsafe immutable {label}", error=V9ArtifactError)
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V9ArtifactError(f"malformed {label}") from exc
    require(isinstance(parsed, dict) and canonical_bytes(parsed) == raw, f"noncanonical {label}", error=V9ArtifactError)
    return parsed


@dataclass(frozen=True, order=True)
class V9Session:
    """A synthetic stand-in for one frozen external session row."""

    asset_id: str
    session_id: str
    query_window_count: int

    def __post_init__(self) -> None:
        _safe_component(self.asset_id, "asset_id")
        _safe_component(self.session_id, "session_id")
        require(isinstance(self.query_window_count, int) and self.query_window_count > 0, "query_window_count must be positive", error=V9Error)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class V9CellKey:
    asset_id: str
    session_id: str
    view: str
    arm: str
    seed: int

    def __post_init__(self) -> None:
        _safe_component(self.asset_id, "asset_id")
        _safe_component(self.session_id, "session_id")
        require(self.view in VIEWS, "V9 view drift", error=V9Error)
        require(self.arm in ARMS, "V9 arm drift", error=V9Error)
        require(self.seed in SEEDS, "V9 seed drift", error=V9Error)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def artifact_relative_path(self) -> str:
        return f"artifacts/{self.asset_id}/{self.view}/{self.arm}/seed_{self.seed}/predictions_targets.npz"

    @property
    def commit_relative_path(self) -> str:
        return f"commits/{self.asset_id}/{self.view}/{self.arm}/seed_{self.seed}.json"


@dataclass(frozen=True)
class V9MatrixContract:
    """Exact synthetic representation of the frozen V9 scientific topology."""

    cohort: tuple[V9Session, ...]
    bootstrap_replicates: int = DEFAULT_SYNTHETIC_BOOTSTRAP_REPLICATES
    bootstrap_seed: int = 68820260805

    def __post_init__(self) -> None:
        require(len(self.cohort) == EXPECTED_SESSION_COUNT, "V9 requires exactly 15 sessions", error=V9Error)
        require(len({row.asset_id for row in self.cohort}) == EXPECTED_SESSION_COUNT, "duplicate V9 asset_id", error=V9Error)
        require(len({row.session_id for row in self.cohort}) == EXPECTED_SESSION_COUNT, "duplicate V9 session_id", error=V9Error)
        require(isinstance(self.bootstrap_replicates, int) and self.bootstrap_replicates > 0, "bootstrap_replicates must be positive", error=V9Error)
        require(isinstance(self.bootstrap_seed, int) and self.bootstrap_seed >= 0, "bootstrap_seed must be nonnegative", error=V9Error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "dandi_000688_subm_co_three_arm_v9_synthetic_contract_v1",
            "scope": SYNTHETIC_SCOPE,
            "cohort": [row.as_dict() for row in self.cohort],
            "views": list(VIEWS),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "expected_cell_count": EXPECTED_CELL_COUNT,
            "chronology": {
                "activity_identity_trials": ACTIVITY_IDENTITY_TRIALS,
                "t4_fit_pool_trials": T4_FIT_POOL_TRIALS,
                "query_rule": QUERY_RULE,
                "chronological": True,
            },
            "bootstrap": {
                "replicates": self.bootstrap_replicates,
                "seed": self.bootstrap_seed,
                "method": "hierarchical_session_then_seed_percentile_linear",
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    @property
    def expected_keys(self) -> tuple[V9CellKey, ...]:
        keys = tuple(
            V9CellKey(row.asset_id, row.session_id, view, arm, seed)
            for row in self.cohort
            for view in VIEWS
            for arm in ARMS
            for seed in SEEDS
        )
        require(len(keys) == EXPECTED_CELL_COUNT and len(set(keys)) == EXPECTED_CELL_COUNT, "V9 exact 270-cell topology drift", error=V9Error)
        return keys

    def query_count_for(self, asset_id: str) -> int:
        matches = [row.query_window_count for row in self.cohort if row.asset_id == asset_id]
        require(len(matches) == 1, "unknown V9 asset", error=V9ArtifactError)
        return matches[0]


@dataclass(frozen=True)
class V9SyntheticRunToken:
    """A non-cryptographic test token that proves execution ordering only.

    It deliberately carries ``formal_external_scoring_permitted=False``.  No
    downstream code should interpret it as an authorization for a real dataset.
    """

    contract_sha256: str
    output_root: str
    purpose: str
    token_id: str
    issued_at: str
    expires_at: str
    formal_external_scoring_permitted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V9RecoveryBinding:
    contract_sha256: str
    manifest_sha256: str
    committed_map_sha256: str
    completed_cell_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V9ArtifactState:
    contract_sha256: str
    manifest_sha256: str
    completed: tuple[V9CellKey, ...]
    missing: tuple[V9CellKey, ...]
    commit_sha256_by_key: Mapping[V9CellKey, str]
    commits_by_key: Mapping[V9CellKey, Mapping[str, Any]]

    @property
    def complete(self) -> bool:
        return len(self.completed) == EXPECTED_CELL_COUNT and not self.missing

    @property
    def committed_map_sha256(self) -> str:
        rows = [
            {"cell": key.as_dict(), "commit_sha256": self.commit_sha256_by_key[key]}
            for key in self.completed
        ]
        return canonical_sha256(rows)

    def recovery_binding(self) -> V9RecoveryBinding:
        return V9RecoveryBinding(
            contract_sha256=self.contract_sha256,
            manifest_sha256=self.manifest_sha256,
            committed_map_sha256=self.committed_map_sha256,
            completed_cell_count=len(self.completed),
        )


def assert_authorization_first_import_boundary() -> None:
    """Fail if a scoring/data runtime was already imported.

    This is intentionally an ordering assertion, not an authorization system.
    It is exercised in a fresh subprocess by the focused tests.  A future
    formal runner must call an equivalent check before importing its runtime.
    """

    forbidden_prefixes = (
        "numpy",
        "torch",
        "pynwb",
        "sua_exploration.mc_maze.multisession_datamodule",
        "sua_exploration.mc_maze.datamodule",
        "sua_exploration.mc_maze.unit_side_features",
        "mc_maze.multisession_datamodule",
        "mc_maze.datamodule",
        "mc_maze.unit_side_features",
    )
    loaded = tuple(sys.modules)
    bad = sorted(
        module
        for module in loaded
        if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden_prefixes)
    )
    require(not bad, f"V9 runtime imported before ordering token: {bad}", error=V9AuthorizationOrderError)


def prepare_synthetic_run_token(
    contract: V9MatrixContract,
    output_root: Path,
    *,
    purpose: str = "initial",
    token_id: str = "synthetic-v9-run",
    now: datetime | None = None,
    validity_seconds: int = 900,
) -> V9SyntheticRunToken:
    """Create a token for synthetic artifact tests only.

    This function intentionally does not call the import-boundary assertion:
    in unit tests NumPy may have been imported by an earlier test.  The caller
    that needs to demonstrate ordering calls
    :func:`assert_authorization_first_import_boundary` in a fresh process
    immediately before this function.  That makes the limitation explicit and
    avoids presenting a mutable Python object as a formal capability.
    """

    require(purpose in {"initial", "recovery"}, "unsupported synthetic token purpose", error=V9AuthorizationOrderError)
    require(isinstance(token_id, str) and _COMPONENT_PATTERN.fullmatch(token_id) is not None, "unsafe token_id", error=V9AuthorizationOrderError)
    require(isinstance(validity_seconds, int) and 1 <= validity_seconds <= 900, "synthetic token validity must be 1..900 seconds", error=V9AuthorizationOrderError)
    issued = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = issued + timedelta(seconds=validity_seconds)
    return V9SyntheticRunToken(
        contract_sha256=contract.sha256,
        output_root=str(output_root.resolve()),
        purpose=purpose,
        token_id=token_id,
        issued_at=issued.isoformat(),
        expires_at=expires.isoformat(),
        formal_external_scoring_permitted=False,
    )


def _validate_synthetic_token(token: V9SyntheticRunToken, contract: V9MatrixContract, output_root: Path, *, purpose: str) -> None:
    require(isinstance(token, V9SyntheticRunToken), "V9 artifact writer requires a synthetic ordering token", error=V9AuthorizationOrderError)
    require(token.purpose == purpose, "synthetic token purpose drift", error=V9AuthorizationOrderError)
    require(token.contract_sha256 == contract.sha256, "synthetic token contract drift", error=V9AuthorizationOrderError)
    require(token.output_root == str(output_root.resolve()), "synthetic token output-root drift", error=V9AuthorizationOrderError)
    require(token.formal_external_scoring_permitted is False, "synthetic token cannot authorize formal scoring", error=V9AuthorizationOrderError)
    try:
        issued = datetime.fromisoformat(token.issued_at)
        expires = datetime.fromisoformat(token.expires_at)
    except ValueError as exc:
        raise V9AuthorizationOrderError("synthetic token time malformed") from exc
    require(issued.tzinfo is not None and expires.tzinfo is not None and expires > issued, "synthetic token time drift", error=V9AuthorizationOrderError)


def _np() -> Any:
    """Import NumPy only after a synthetic ordering token reached runtime."""

    import numpy as np

    return np


def _validate_arrays(predictions: Any, targets: Any, *, expected_rows: int) -> tuple[Any, Any]:
    np = _np()
    require(isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray), "prediction/target must be NumPy arrays", error=V9ArtifactError)
    require(predictions.dtype == np.dtype("float32") and targets.dtype == np.dtype("float32"), "prediction/target must be exact float32", error=V9ArtifactError)
    require(predictions.shape == targets.shape == (expected_rows, OUTPUT_DIM), "prediction/target shape drift", error=V9ArtifactError)
    require(predictions.flags.c_contiguous and targets.flags.c_contiguous, "prediction/target must be C-contiguous", error=V9ArtifactError)
    require(bool(np.isfinite(predictions).all()) and bool(np.isfinite(targets).all()), "prediction/target contains NaN/Inf", error=V9ArtifactError)
    return predictions, targets


def query_behavior_trace_sha256(targets: Any) -> str:
    """Hash the ordered query-behavior target trace without scoring it.

    The row order is the query-index order.  The header fixes that convention,
    the exact float32 shape, and the binary representation before the ordered
    target bytes are hashed.  It is therefore a Phase-A pairing receipt, not a
    metric: the same held-out asset/session must carry the exact same receipt
    in every view, arm, and seed.
    """

    np = _np()
    require(isinstance(targets, np.ndarray), "query behavior targets must be a NumPy array", error=V9ArtifactError)
    require(targets.dtype == np.dtype("float32"), "query behavior targets must be exact float32", error=V9ArtifactError)
    require(targets.ndim == 2 and targets.shape[1] == OUTPUT_DIM, "query behavior target shape drift", error=V9ArtifactError)
    require(targets.flags.c_contiguous, "query behavior targets must be C-contiguous", error=V9ArtifactError)
    require(bool(np.isfinite(targets).all()), "query behavior targets contain NaN/Inf", error=V9ArtifactError)
    digest = hashlib.sha256()
    digest.update(
        canonical_bytes(
            {
                "schema": "dandi_000688_subm_co_three_arm_v9_query_behavior_trace_v1",
                "dtype": "float32",
                "shape": [int(targets.shape[0]), OUTPUT_DIM],
                "query_index_order": "ascending_zero_based_row_order",
            }
        )
    )
    digest.update(targets.tobytes(order="C"))
    return digest.hexdigest()


def _require_query_behavior_trace_evidence(evidence: Mapping[str, Any], targets: Any) -> str:
    """Require an evidence trace and bind it to the actual target bytes."""

    trace = evidence.get("query_behavior_trace_sha256")
    require(
        isinstance(trace, str) and re.fullmatch(r"[0-9a-f]{64}", trace) is not None,
        "evidence requires a lowercase query_behavior_trace_sha256",
        error=V9ArtifactError,
    )
    require(
        trace == query_behavior_trace_sha256(targets),
        "query_behavior_trace_sha256 does not match ordered artifact targets",
        error=V9ArtifactError,
    )
    return trace


def _asset_session_identity(key: V9CellKey) -> tuple[str, str]:
    return key.asset_id, key.session_id


def _npz_max_bytes(expected_rows: int) -> int:
    # The raw payload is two [Q,2] float32 arrays: 16*Q bytes.  The cap gives
    # ordinary compressed-NPZ overhead room without accepting unbounded input.
    return max(1024 * 1024, expected_rows * OUTPUT_DIM * 4 * 2 * 16 + 1024 * 1024)


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
    require(_is_readonly_regular(path), f"immutable NPZ write failed: {path.name}", error=V9ArtifactError)
    return sha256_file(path)


def _read_npz(
    path: Path,
    *,
    expected_rows: int,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[Any, Any]:
    np = _np()
    require(_is_readonly_regular(path), "NPZ is missing, mutable, or symlinked", error=V9ArtifactError)
    metadata = path.stat()
    require(metadata.st_size <= _npz_max_bytes(expected_rows), "NPZ exceeds fixed synthetic safety cap", error=V9ArtifactError)
    if expected_sha256 is not None:
        require(sha256_file(path) == expected_sha256, "NPZ SHA-256 drift", error=V9ArtifactError)
    if expected_bytes is not None:
        require(metadata.st_size == expected_bytes, "NPZ byte count drift", error=V9ArtifactError)
    try:
        with np.load(path, allow_pickle=False) as loaded:
            require(tuple(loaded.files) == ("predictions", "targets"), "NPZ member order/key set drift", error=V9ArtifactError)
            predictions = np.ascontiguousarray(loaded["predictions"])
            targets = np.ascontiguousarray(loaded["targets"])
    except (OSError, ValueError) as exc:
        raise V9ArtifactError("cannot safely open immutable NPZ") from exc
    return _validate_arrays(predictions, targets, expected_rows=expected_rows)


def _contains_forbidden_metric_key(value: Any) -> bool:
    forbidden = {"r2", "score", "metric", "delta", "gate", "performance", "accuracy"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in forbidden:
                return True
            if _contains_forbidden_metric_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_metric_key(item) for item in value)
    return False


def _list_regular_files(root: Path) -> set[str]:
    require(root.is_dir() and not root.is_symlink(), "artifact root missing or unsafe", error=V9ArtifactError)
    result: set[str] = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            entry = directory_path / name
            require(not entry.is_symlink(), "symlink in artifact directory topology", error=V9ArtifactError)
        for name in filenames:
            entry = directory_path / name
            require(not entry.is_symlink() and entry.is_file(), "non-regular artifact entry", error=V9ArtifactError)
            result.add(str(entry.relative_to(root)))
    return result


def _manifest_payload(contract: V9MatrixContract, token: V9SyntheticRunToken) -> dict[str, Any]:
    return {
        "schema": "dandi_000688_subm_co_three_arm_v9_synthetic_run_manifest_v1",
        "status": "ARTIFACT_ONLY_SYNTHETIC_NOT_FORMAL",
        "scope": SYNTHETIC_SCOPE,
        "contract": contract.as_dict(),
        "contract_sha256": contract.sha256,
        "token": token.as_dict(),
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "phase_a_metrics": "FORBIDDEN_UNTIL_EXACT_270_ARTIFACT_COMMITS",
        "no_overwrite": True,
        "missing_only_continuation": True,
        "external_subm_opened": False,
        "checkpoint_opened": False,
        "model_forward_performed": False,
    }


def _parse_cell_key(value: Mapping[str, Any]) -> V9CellKey:
    require(set(value) == {"asset_id", "session_id", "view", "arm", "seed"}, "cell key schema drift", error=V9ArtifactError)
    return V9CellKey(
        asset_id=str(value["asset_id"]),
        session_id=str(value["session_id"]),
        view=str(value["view"]),
        arm=str(value["arm"]),
        seed=int(value["seed"]),
    )


class V9ArtifactOnlyWriter:
    """Write synthetic Phase-A artifacts; never calculate a metric.

    The writer accepts only a synthetic ordering token and therefore cannot be
    accidentally used as an external scorer.  A later formal executor should
    retain the artifact/commit semantics but replace the token with a reviewed
    authorization layer.
    """

    def __init__(self, output_root: Path, contract: V9MatrixContract, token: V9SyntheticRunToken) -> None:
        self.root = output_root.resolve()
        _validate_synthetic_token(token, contract, self.root, purpose="initial")
        require(not self.root.exists(), "artifact output root already exists; overwrite forbidden", error=V9ArtifactError)
        self.root.mkdir(parents=True, exist_ok=False)
        self.contract = contract
        self.token = token
        self._expected = set(contract.expected_keys)
        self._written: set[V9CellKey] = set()
        self._behavior_trace_by_asset_session: dict[tuple[str, str], str] = {}
        self._behavior_target_bytes_by_asset_session: dict[tuple[str, str], bytes] = {}
        payload = _manifest_payload(contract, token)
        self._manifest_path = self.root / "run_manifest.json"
        self._manifest_sha256 = _write_json_exclusive(self._manifest_path, payload)

    @classmethod
    def open_missing_only_recovery(
        cls,
        output_root: Path,
        contract: V9MatrixContract,
        token: V9SyntheticRunToken,
        binding: V9RecoveryBinding,
    ) -> "V9ArtifactOnlyWriter":
        root = output_root.resolve()
        _validate_synthetic_token(token, contract, root, purpose="recovery")
        state = scan_artifact_state(root, contract)
        observed = state.recovery_binding()
        require(observed == binding, "recovery binding differs from exact artifact state", error=V9ContinuationError)
        instance = object.__new__(cls)
        instance.root = root
        instance.contract = contract
        instance.token = token
        instance._expected = set(contract.expected_keys)
        instance._written = set(state.completed)
        instance._behavior_trace_by_asset_session = {}
        instance._behavior_target_bytes_by_asset_session = {}
        instance._manifest_path = root / "run_manifest.json"
        instance._manifest_sha256 = state.manifest_sha256
        for key in state.completed:
            commit = state.commits_by_key[key]
            artifact = commit["prediction_target_artifact"]
            _prediction, targets = _read_npz(
                _safe_relative(root, key.artifact_relative_path, label="artifact"),
                expected_rows=contract.query_count_for(key.asset_id),
                expected_sha256=str(artifact["sha256"]),
                expected_bytes=int(artifact["bytes"]),
            )
            trace = str(commit["evidence"]["query_behavior_trace_sha256"])
            instance._require_paired_behavior_target(key, targets, trace)
            instance._record_paired_behavior_target(key, targets, trace)
        return instance

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    def _require_paired_behavior_target(self, key: V9CellKey, targets: Any, trace: str) -> None:
        """Reject a target mismatch before it can create an artifact commit."""

        identity = _asset_session_identity(key)
        prior_trace = self._behavior_trace_by_asset_session.get(identity)
        prior_bytes = self._behavior_target_bytes_by_asset_session.get(identity)
        if prior_trace is None:
            require(prior_bytes is None, "writer behavior-pairing state drift", error=V9ArtifactError)
            return
        require(prior_bytes is not None, "writer behavior-pairing state drift", error=V9ArtifactError)
        require(
            trace == prior_trace,
            "query behavior trace differs within one asset/session across V9 cells",
            error=V9ArtifactError,
        )
        require(
            targets.tobytes(order="C") == prior_bytes,
            "behavior targets must be bitwise identical within one asset/session across V9 cells",
            error=V9ArtifactError,
        )

    def _record_paired_behavior_target(self, key: V9CellKey, targets: Any, trace: str) -> None:
        identity = _asset_session_identity(key)
        self._behavior_trace_by_asset_session.setdefault(identity, trace)
        self._behavior_target_bytes_by_asset_session.setdefault(identity, targets.tobytes(order="C"))

    def write_cell_artifact(
        self,
        key: V9CellKey,
        predictions: Any,
        targets: Any,
        *,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        require(key in self._expected, "cell is outside exact V9 270-cell contract", error=V9ArtifactError)
        require(key not in self._written, "cell already committed; recomputation/overwrite forbidden", error=V9ContinuationError)
        require(isinstance(evidence, Mapping) and not _contains_forbidden_metric_key(evidence), "Phase-A evidence cannot carry a metric/result field", error=V9ArtifactError)
        rows = self.contract.query_count_for(key.asset_id)
        predictions, targets = _validate_arrays(predictions, targets, expected_rows=rows)
        trace = _require_query_behavior_trace_evidence(evidence, targets)
        self._require_paired_behavior_target(key, targets, trace)
        artifact_path = _safe_relative(self.root, key.artifact_relative_path, label="artifact")
        commit_path = _safe_relative(self.root, key.commit_relative_path, label="commit")
        artifact_sha = _write_npz_exclusive(artifact_path, predictions, targets)
        reopened_prediction, reopened_target = _read_npz(
            artifact_path,
            expected_rows=rows,
            expected_sha256=artifact_sha,
            expected_bytes=artifact_path.stat().st_size,
        )
        require(reopened_prediction.shape == predictions.shape and reopened_target.shape == targets.shape, "artifact reopen shape drift", error=V9ArtifactError)
        require(
            query_behavior_trace_sha256(reopened_target) == trace,
            "artifact reopen changed the query behavior target trace",
            error=V9ArtifactError,
        )
        payload = {
            "schema": "dandi_000688_subm_co_three_arm_v9_synthetic_artifact_commit_v1",
            "status": "ARTIFACT_COMMITTED_NO_METRIC",
            "scope": SYNTHETIC_SCOPE,
            "contract_sha256": self.contract.sha256,
            "cell": key.as_dict(),
            "query_window_count": rows,
            "prediction_target_artifact": {
                "relative_path": key.artifact_relative_path,
                "sha256": artifact_sha,
                "bytes": artifact_path.stat().st_size,
                "mode": "0444",
                "dtype": "float32",
                "shape": [rows, OUTPUT_DIM],
                "arrays": ["predictions", "targets"],
            },
            "evidence": dict(evidence),
            "metric_computation": "FORBIDDEN_IN_PHASE_A",
        }
        require(not _contains_forbidden_metric_key(payload["evidence"]), "Phase-A evidence acquired metric content", error=V9ArtifactError)
        commit_sha = _write_json_exclusive(commit_path, payload)
        self._written.add(key)
        self._record_paired_behavior_target(key, targets, trace)
        return {
            "cell": key.as_dict(),
            "artifact_relative_path": key.artifact_relative_path,
            "artifact_sha256": artifact_sha,
            "commit_relative_path": key.commit_relative_path,
            "commit_sha256": commit_sha,
            "metric_computed": False,
        }

    def state(self) -> V9ArtifactState:
        return scan_artifact_state(self.root, self.contract)


def _validate_manifest(root: Path, contract: V9MatrixContract) -> tuple[dict[str, Any], str]:
    path = _safe_relative(root, "run_manifest.json", label="run manifest")
    payload = _read_canonical_json(path, label="run manifest")
    expected_keys = {
        "schema", "status", "scope", "contract", "contract_sha256", "token", "expected_cell_count",
        "phase_a_metrics", "no_overwrite", "missing_only_continuation", "external_subm_opened",
        "checkpoint_opened", "model_forward_performed",
    }
    require(set(payload) == expected_keys, "run manifest schema drift", error=V9ArtifactError)
    require(payload["schema"] == "dandi_000688_subm_co_three_arm_v9_synthetic_run_manifest_v1", "run manifest schema mismatch", error=V9ArtifactError)
    require(payload["status"] == "ARTIFACT_ONLY_SYNTHETIC_NOT_FORMAL" and payload["scope"] == SYNTHETIC_SCOPE, "run manifest scope/status drift", error=V9ArtifactError)
    require(payload["contract"] == contract.as_dict() and payload["contract_sha256"] == contract.sha256, "run manifest contract drift", error=V9ArtifactError)
    require(payload["expected_cell_count"] == EXPECTED_CELL_COUNT, "run manifest cardinality drift", error=V9ArtifactError)
    require(payload["phase_a_metrics"] == "FORBIDDEN_UNTIL_EXACT_270_ARTIFACT_COMMITS", "run manifest partial-metric policy drift", error=V9ArtifactError)
    require(payload["no_overwrite"] is True and payload["missing_only_continuation"] is True, "run manifest continuation policy drift", error=V9ArtifactError)
    require(payload["external_subm_opened"] is False and payload["checkpoint_opened"] is False and payload["model_forward_performed"] is False, "synthetic manifest claims external runtime", error=V9ArtifactError)
    require(isinstance(payload["token"], Mapping) and payload["token"].get("formal_external_scoring_permitted") is False, "run manifest token drift", error=V9ArtifactError)
    return payload, sha256_file(path)


def _validate_commit(
    root: Path,
    contract: V9MatrixContract,
    key: V9CellKey,
) -> tuple[dict[str, Any], str, Any]:
    commit_path = _safe_relative(root, key.commit_relative_path, label="commit")
    payload = _read_canonical_json(commit_path, label="artifact commit")
    expected_keys = {
        "schema", "status", "scope", "contract_sha256", "cell", "query_window_count",
        "prediction_target_artifact", "evidence", "metric_computation",
    }
    require(set(payload) == expected_keys, "artifact commit schema drift", error=V9ArtifactError)
    require(payload["schema"] == "dandi_000688_subm_co_three_arm_v9_synthetic_artifact_commit_v1", "artifact commit schema mismatch", error=V9ArtifactError)
    require(payload["status"] == "ARTIFACT_COMMITTED_NO_METRIC" and payload["scope"] == SYNTHETIC_SCOPE, "artifact commit scope/status drift", error=V9ArtifactError)
    require(payload["contract_sha256"] == contract.sha256 and _parse_cell_key(payload["cell"]) == key, "artifact commit identity drift", error=V9ArtifactError)
    rows = contract.query_count_for(key.asset_id)
    require(payload["query_window_count"] == rows, "artifact commit query count drift", error=V9ArtifactError)
    require(payload["metric_computation"] == "FORBIDDEN_IN_PHASE_A", "artifact commit metric policy drift", error=V9ArtifactError)
    require(isinstance(payload["evidence"], Mapping) and not _contains_forbidden_metric_key(payload["evidence"]), "artifact commit contains metric/result evidence", error=V9ArtifactError)
    artifact = payload["prediction_target_artifact"]
    expected_artifact_keys = {"relative_path", "sha256", "bytes", "mode", "dtype", "shape", "arrays"}
    require(isinstance(artifact, Mapping) and set(artifact) == expected_artifact_keys, "artifact pin schema drift", error=V9ArtifactError)
    require(artifact["relative_path"] == key.artifact_relative_path and artifact["mode"] == "0444", "artifact relative path/mode drift", error=V9ArtifactError)
    require(artifact["dtype"] == "float32" and artifact["shape"] == [rows, OUTPUT_DIM] and artifact["arrays"] == ["predictions", "targets"], "artifact array schema drift", error=V9ArtifactError)
    require(isinstance(artifact["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None, "artifact SHA malformed", error=V9ArtifactError)
    require(isinstance(artifact["bytes"], int) and artifact["bytes"] > 0, "artifact bytes malformed", error=V9ArtifactError)
    artifact_path = _safe_relative(root, key.artifact_relative_path, label="artifact")
    _prediction, targets = _read_npz(
        artifact_path,
        expected_rows=rows,
        expected_sha256=str(artifact["sha256"]),
        expected_bytes=int(artifact["bytes"]),
    )
    _require_query_behavior_trace_evidence(payload["evidence"], targets)
    return payload, sha256_file(commit_path), targets


def scan_artifact_state(root: Path, contract: V9MatrixContract) -> V9ArtifactState:
    """Verify topology/arrays/commits without calculating any metric."""

    root = root.resolve()
    _manifest, manifest_sha = _validate_manifest(root, contract)
    expected = contract.expected_keys
    expected_artifacts = {key.artifact_relative_path for key in expected}
    expected_commits = {key.commit_relative_path for key in expected}
    observed = _list_regular_files(root)
    allowed = {"run_manifest.json"} | expected_artifacts | expected_commits
    require(observed <= allowed, "unknown artifact topology entry", error=V9ArtifactError)
    complete: list[V9CellKey] = []
    missing: list[V9CellKey] = []
    commit_sha: dict[V9CellKey, str] = {}
    commits: dict[V9CellKey, Mapping[str, Any]] = {}
    behavior_trace_by_asset_session: dict[tuple[str, str], str] = {}
    behavior_target_bytes_by_asset_session: dict[tuple[str, str], bytes] = {}
    for key in expected:
        artifact_present = key.artifact_relative_path in observed
        commit_present = key.commit_relative_path in observed
        require(artifact_present == commit_present, "orphan artifact or orphan commit", error=V9ArtifactError)
        if not artifact_present:
            missing.append(key)
            continue
        payload, digest, targets = _validate_commit(root, contract, key)
        identity = _asset_session_identity(key)
        trace = str(payload["evidence"]["query_behavior_trace_sha256"])
        target_bytes = targets.tobytes(order="C")
        prior_trace = behavior_trace_by_asset_session.get(identity)
        prior_target_bytes = behavior_target_bytes_by_asset_session.get(identity)
        if prior_trace is None:
            require(prior_target_bytes is None, "scan behavior-pairing state drift", error=V9ArtifactError)
            behavior_trace_by_asset_session[identity] = trace
            behavior_target_bytes_by_asset_session[identity] = target_bytes
        else:
            require(prior_target_bytes is not None, "scan behavior-pairing state drift", error=V9ArtifactError)
            require(
                trace == prior_trace,
                "query behavior trace differs within one asset/session across V9 cells",
                error=V9ArtifactError,
            )
            require(
                target_bytes == prior_target_bytes,
                "behavior targets must be bitwise identical within one asset/session across V9 cells",
                error=V9ArtifactError,
            )
        complete.append(key)
        commit_sha[key] = digest
        commits[key] = payload
    return V9ArtifactState(
        contract_sha256=contract.sha256,
        manifest_sha256=manifest_sha,
        completed=tuple(complete),
        missing=tuple(missing),
        commit_sha256_by_key=commit_sha,
        commits_by_key=commits,
    )


def _synthetic_variance_weighted_r2(prediction: Any, target: Any) -> float:
    """CPU-only finite R² for *synthetic* finalizer tests.

    This is intentionally not the formal TorchMetrics binding.  A future
    formal successor must use the separately vetted frozen CPU TorchMetrics
    re-opener.  Keeping this implementation local prevents the synthetic
    pre-execution module from importing the old external scoring stack.
    """

    np = _np()
    prediction, target = _validate_arrays(prediction, target, expected_rows=int(target.shape[0]))
    require(prediction.shape[0] >= 2, "synthetic R² needs at least two rows", error=V9ArtifactError)
    target64 = target.astype(np.float64, copy=False)
    pred64 = prediction.astype(np.float64, copy=False)
    centered = target64 - target64.mean(axis=0, keepdims=True)
    tss = np.sum(centered * centered, axis=0, dtype=np.float64)
    rss = np.sum((target64 - pred64) ** 2, axis=0, dtype=np.float64)
    require(bool(np.all(tss > 0)), "synthetic R² target is constant", error=V9ArtifactError)
    raw = 1.0 - rss / tss
    value = float(np.sum(tss * raw, dtype=np.float64) / np.sum(tss, dtype=np.float64))
    require(math.isfinite(value), "synthetic R² is nonfinite", error=V9ArtifactError)
    return value


def _bootstrap(delta: Any, *, replicates: int, seed: int) -> tuple[float, float]:
    np = _np()
    require(delta.shape == (EXPECTED_SESSION_COUNT, len(SEEDS)), "bootstrap delta shape drift", error=V9ArtifactError)
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 10_000):
        count = min(10_000, replicates - start)
        sessions = rng.integers(0, EXPECTED_SESSION_COUNT, size=(count, EXPECTED_SESSION_COUNT))
        seeds = rng.integers(0, len(SEEDS), size=(count, EXPECTED_SESSION_COUNT, len(SEEDS)))
        values[start:start + count] = delta[sessions[:, :, None], seeds].mean(axis=(1, 2))
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def _comparison_summary(
    t4: Any,
    other: Any,
    *,
    contract: V9MatrixContract,
    comparison_index: int,
) -> dict[str, Any]:
    np = _np()
    require(t4.shape == other.shape == (EXPECTED_SESSION_COUNT, len(SEEDS)), "comparison grid shape drift", error=V9ArtifactError)
    delta = t4 - other
    seed_means = delta.mean(axis=0)
    session_means = delta.mean(axis=1)
    lower, upper = _bootstrap(
        delta,
        replicates=contract.bootstrap_replicates,
        seed=contract.bootstrap_seed + comparison_index,
    )
    grand = float(delta.mean())
    t4_seed_means = t4.mean(axis=0)
    gates = {
        "grand_paired_mean_at_least_0p03": grand >= 0.03,
        "all_three_seed_means_strictly_positive": bool(np.all(seed_means > 0)),
        "at_least_12_of_15_session_means_positive": int(np.sum(session_means > 0)) >= 12,
        "hierarchical_bootstrap_lower_95_strictly_positive": lower > 0,
        "shared_t4_absolute_grand_and_seed_means_strictly_positive": bool(t4.mean() > 0 and np.all(t4_seed_means > 0)),
    }
    return {
        "grand_paired_mean_r2": grand,
        "seed_means_r2": {str(seed): float(seed_means[index]) for index, seed in enumerate(SEEDS)},
        "session_cross_seed_means_r2": {
            contract.cohort[index].session_id: float(session_means[index])
            for index in range(EXPECTED_SESSION_COUNT)
        },
        "positive_session_count": int(np.sum(session_means > 0)),
        "positive_session_required": 12,
        "hierarchical_bootstrap_95": {"lower": lower, "upper": upper},
        "shared_t4_absolute_grand_r2": float(t4.mean()),
        "shared_t4_absolute_seed_means_r2": {str(seed): float(t4_seed_means[index]) for index, seed in enumerate(SEEDS)},
        "gates": gates,
        "view_pass": bool(all(gates.values())),
    }


def finalize_synthetic_artifacts(root: Path, contract: V9MatrixContract) -> dict[str, Any]:
    """Open the synthetic metric phase only after exact 270 artifact commits."""

    root = root.resolve()
    aggregate_path = _safe_relative(root, "aggregate/endpoint_aggregate.json", label="aggregate")
    require(not aggregate_path.exists(), "aggregate overwrite/retry forbidden", error=V9ArtifactError)
    state = scan_artifact_state(root, contract)
    require(state.complete, "finalizer is forbidden before exact 270 artifact commits", error=V9IncompleteMatrixError)
    np = _np()
    grids = {
        view: {arm: np.empty((EXPECTED_SESSION_COUNT, len(SEEDS)), dtype=np.float64) for arm in ARMS}
        for view in VIEWS
    }
    cells: list[dict[str, Any]] = []
    for session_index, session in enumerate(contract.cohort):
        for view in VIEWS:
            for arm in ARMS:
                for seed_index, seed in enumerate(SEEDS):
                    key = V9CellKey(session.asset_id, session.session_id, view, arm, seed)
                    commit = state.commits_by_key[key]
                    artifact = commit["prediction_target_artifact"]
                    prediction, target = _read_npz(
                        _safe_relative(root, key.artifact_relative_path, label="artifact"),
                        expected_rows=session.query_window_count,
                        expected_sha256=str(artifact["sha256"]),
                        expected_bytes=int(artifact["bytes"]),
                    )
                    value = _synthetic_variance_weighted_r2(prediction, target)
                    grids[view][arm][session_index, seed_index] = value
                    cells.append({"cell": key.as_dict(), "r2": value})
    comparisons: dict[str, Any] = {}
    for comparison_index, (name, comparator) in enumerate(
        (("shared_t4_minus_shared_zero4", "shared_zero4"), ("shared_t4_minus_shared_ts4", "shared_ts4"))
    ):
        views = {
            view: _comparison_summary(
                grids[view]["shared_t4"],
                grids[view][comparator],
                contract=contract,
                comparison_index=comparison_index * len(VIEWS) + view_index,
            )
            for view_index, view in enumerate(VIEWS)
        }
        comparisons[name] = {
            "views": views,
            "comparison_pass": bool(all(views[view]["view_pass"] for view in VIEWS)),
            "cross_view_rescue_used": False,
        }
    payload = {
        "schema": "dandi_000688_subm_co_three_arm_v9_synthetic_final_aggregate_v1",
        "status": "SYNTHETIC_FULL_270_FINALIZED_NOT_FORMAL",
        "scope": SYNTHETIC_SCOPE,
        "contract_sha256": contract.sha256,
        "manifest_sha256": state.manifest_sha256,
        "verified_cell_count": EXPECTED_CELL_COUNT,
        "metric_definition": "synthetic_cpu_variance_weighted_r2_not_formal_torchmetrics",
        "cells": cells,
        "comparisons": comparisons,
        "primary_sua_absolute_pass": comparisons["shared_t4_minus_shared_zero4"]["views"]["sua"]["view_pass"],
        "key_secondary_pseudo_mua_absolute_pass": comparisons["shared_t4_minus_shared_zero4"]["views"]["pseudo_mua"]["view_pass"],
        "overall_three_arm_claim_pass": bool(all(row["comparison_pass"] for row in comparisons.values())),
        "aggregate_recomputed_only_after_full_artifact_matrix": True,
    }
    aggregate_sha = _write_json_exclusive(aggregate_path, payload)
    return {"aggregate": payload, "aggregate_sha256": aggregate_sha}


def direct_zero4_from_channel_count(channel_count: int) -> tuple[Any, dict[str, Any]]:
    """Construct direct standardized zero4 from **only** the channel count."""

    np = _np()
    require(not isinstance(channel_count, bool) and isinstance(channel_count, (int, np.integer)), "zero4 channel_count must be an integer", error=V9Zero4Error)
    count = int(channel_count)
    require(0 < count < 100, "zero4 channel_count must satisfy 0 < channel_count < 100", error=V9Zero4Error)
    values = np.zeros((count, 4), dtype=np.float32)
    require_direct_zero4(values)
    receipt = {
        "construction_input": ["channel_count"],
        "target_direction_label_reads_for_descriptor": 0,
        "t4_trial_rate_reads_for_descriptor": 0,
        "target_t4_rate_fit_calls": 0,
        "source_t4_normalizer_value_reads": 0,
        "source_t4_normalizer_arithmetic_performed": False,
        "raw_t4_constructed": False,
        "side_feature_loader_calls_for_zero4": 0,
        "side_normalizer_passed_to_zero4_constructor": False,
        "bitwise_positive_float32_zero": True,
        "side_shape": [count, 4],
    }
    return values, receipt


def require_direct_zero4(values: Any) -> None:
    np = _np()
    array = np.asarray(values)
    require(
        array.dtype == np.dtype(np.float32)
        and array.ndim == 2
        and array.shape[1] == 4
        and bool(np.all(array.view(np.uint32) == np.uint32(0))),
        "zero4 must be bitwise positive float32 [N,4]",
        error=V9Zero4Error,
    )


def attach_direct_zero4_to_synthetic_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Attach zero4 while intentionally not touching label/rate/normalizer fields.

    The function reads only ``n_units`` and ``neural``.  Tests put poison
    objects under target/rate/normalizer fields to make accidental reads fail.
    """

    require(isinstance(record, Mapping), "zero4 record must be a mapping", error=V9Zero4Error)
    require("n_units" in record and "neural" in record, "zero4 record misses n_units/neural", error=V9Zero4Error)
    np = _np()
    n_units = record["n_units"]
    neural = np.asarray(record["neural"])
    require(not isinstance(n_units, bool) and isinstance(n_units, (int, np.integer)), "zero4 record n_units malformed", error=V9Zero4Error)
    require(neural.ndim == 2 and int(n_units) == int(neural.shape[1]), "zero4 record channel axis drift", error=V9Zero4Error)
    side, receipt = direct_zero4_from_channel_count(int(n_units))
    updated = dict(record)
    updated["side_features"] = side
    updated["zero4_descriptor_receipt"] = receipt
    return updated


def formal_external_execution_unavailable() -> None:
    """Make the absence of an external sub-M runner explicit and testable."""

    raise V9AuthorizationOrderError(
        "V9 synthetic core has no formal authorization, NWB/checkpoint/model/GPU execution path"
    )
