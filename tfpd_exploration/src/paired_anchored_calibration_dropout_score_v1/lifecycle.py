"""Attempt-first immutable lifecycle for the deferred PACD matched scorer.

The public CLI never manufactures a capability.  The only live mint path is
kept here so that it can bind the target-free review graph, two reserved roots,
the closure, and the physical runtime profile *before* target materialization.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from . import plan
from .binding import PACDProducerBinding, verify_historical_comparators, verify_live_v3_producers


class LifecycleError(RuntimeError):
    pass


_SECRET = object()
_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_BUNDLE_NAMES = frozenset({"input_authority.json", "score.json", "terminal.json"})
_FAILURE_NAMES = frozenset({"attempt.json", "failure.json"})


@dataclass(frozen=True)
class ScoreExecutionProfile:
    """Immutable route identity for the one shared score lifecycle.

    The default preserves Score V1.  Successors may only replace producer
    validation and immutable route identities; score science stays in
    :mod:`score` and is deliberately not injected here.
    """

    identity: str
    cell: str
    schema: str
    score_root_relative: str
    authority_root_relative: str
    bound_patterns: tuple[str, ...]
    expected_row_count: int
    binding_type_identity: str
    producer_validator: Callable[[Path, Any], dict]


V1_SCORE_PROFILE = ScoreExecutionProfile(
    identity="pacd-matched-score-v1",
    cell=plan.CELL,
    schema=plan.SCHEMA,
    score_root_relative=plan.RESULT_ROOT_RELATIVE,
    authority_root_relative=plan.AUTHORITY_ROOT_RELATIVE,
    bound_patterns=plan.BOUND_PATTERNS,
    expected_row_count=plan.EXPECTED_ROW_COUNT,
    binding_type_identity="PACDProducerBinding:v1",
    producer_validator=verify_live_v3_producers,
)


def _canonical_relative(value: str, label: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) != value:
        raise LifecycleError(label)
    return value


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise LifecycleError(label)
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def binding_sha256(binding: Any) -> str:
    return _digest(binding.payload())


def _require_binding_type(binding: Any, execution_profile: ScoreExecutionProfile) -> None:
    if getattr(binding, "TYPE_IDENTITY", None) != execution_profile.binding_type_identity:
        raise LifecycleError("binding-type/profile drift")


def source_closure(root: Path, profile: ScoreExecutionProfile = V1_SCORE_PROFILE) -> dict:
    """Exact explicit execution closure; no result traversal or globbing."""
    files: dict[str, dict] = {}
    for relative in profile.bound_patterns:
        path = Path(root) / relative
        if not path.is_file() or path.is_symlink():
            raise LifecycleError(f"closure file drift: {relative}")
        body = path.read_bytes()
        files[relative] = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    digest = _digest(files)
    return {"closure_sha256": digest, "files": files}


@dataclass(frozen=True)
class RootWitness:
    relative: str
    parent_device: int
    parent_inode: int
    name: str
    root_device: int | None = None
    root_inode: int | None = None

    @classmethod
    def fresh(cls, repository_root: Path, relative: str) -> "RootWitness":
        relative = _canonical_relative(relative, "root relative")
        path = Path(repository_root) / relative
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink() or path.exists() or path.is_symlink():
            raise LifecycleError("prospective root is not fresh")
        state = parent.stat(follow_symlinks=False)
        return cls(relative, int(state.st_dev), int(state.st_ino), path.name)

    def verify_parent(self, repository_root: Path) -> Path:
        path = Path(repository_root) / self.relative
        parent = path.parent
        state = parent.stat(follow_symlinks=False)
        if parent.is_symlink() or int(state.st_dev) != self.parent_device or int(state.st_ino) != self.parent_inode or path.name != self.name:
            raise LifecycleError("result-parent identity drift")
        return path

    def verify_reserved(self, repository_root: Path) -> Path:
        path = self.verify_parent(repository_root)
        if not path.is_dir() or path.is_symlink():
            raise LifecycleError("reserved root identity drift")
        state = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(state.st_mode) or self.root_device is None or self.root_inode is None or int(state.st_dev) != self.root_device or int(state.st_ino) != self.root_inode:
            raise LifecycleError("reserved root type")
        return path

    def capture_reserved(self, repository_root: Path) -> "RootWitness":
        path = self.verify_parent(repository_root)
        state = path.stat(follow_symlinks=False)
        if not path.is_dir() or path.is_symlink() or not stat.S_ISDIR(state.st_mode):
            raise LifecycleError("reserved root type")
        return replace(self, root_device=int(state.st_dev), root_inode=int(state.st_ino))


@dataclass(frozen=True)
class BundleWitness:
    parent_device: int
    parent_inode: int
    device: int
    inode: int
    name: str = "complete"

    @classmethod
    def staged(cls, parent: Path, stage: Path) -> "BundleWitness":
        parent_state = parent.stat(follow_symlinks=False)
        stage_state = stage.stat(follow_symlinks=False)
        if stage.is_symlink() or not stat.S_ISDIR(stage_state.st_mode):
            raise LifecycleError("staging bundle type")
        return cls(int(parent_state.st_dev), int(parent_state.st_ino), int(stage_state.st_dev), int(stage_state.st_ino))

    def verify(self, parent: Path) -> Path:
        parent_state = parent.stat(follow_symlinks=False)
        bundle = parent / self.name
        state = bundle.stat(follow_symlinks=False)
        if bundle.is_symlink() or not stat.S_ISDIR(state.st_mode) or (int(parent_state.st_dev), int(parent_state.st_ino), int(state.st_dev), int(state.st_ino)) != (self.parent_device, self.parent_inode, self.device, self.inode):
            raise LifecycleError("committed bundle identity")
        return bundle


class ScoreCapability:
    """Opaque, one-shot in-process authority; no public dataclass constructor."""
    def __init__(self, secret: object, *, binding_digest: str, score_root: RootWitness,
                 authority_root: RootWitness, closure_sha256: str, producer_witness_sha256: str,
                 comparator_witness_sha256: str, authority_sha256: str, runtime_profile: Mapping[str, Any],
                 runtime_attestor: Callable[[], Mapping[str, Any]], mode: str,
                 execution_profile: ScoreExecutionProfile):
        if secret is not _SECRET:
            raise LifecycleError("score capability is nonconstructible")
        self._binding_digest = _sha(binding_digest, "binding digest")
        self._score_root = score_root
        self._authority_root = authority_root
        self._closure_sha256 = _sha(closure_sha256, "closure")
        self._producer_witness_sha256 = _sha(producer_witness_sha256, "producer witness")
        self._comparator_witness_sha256 = _sha(comparator_witness_sha256, "comparator witness")
        self._authority_sha256 = _sha(authority_sha256, "authority receipt")
        self._runtime_profile = dict(runtime_profile)
        self._runtime_attestor = runtime_attestor
        self._mode = mode
        self._execution_profile = execution_profile
        self._profile_identity = execution_profile.identity
        self._binding_type_identity = execution_profile.binding_type_identity
        self._consumed = False


def _descriptor_payload(path: Path) -> tuple[dict, str]:
    """Reload exact immutable receipt body/sidecar, including canonical mode."""
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != _READ_ONLY:
        raise LifecycleError("receipt type/mode")
    side = path.with_suffix(path.suffix + ".sha256")
    if side.is_symlink() or not side.is_file() or stat.S_IMODE(side.stat(follow_symlinks=False).st_mode) != _READ_ONLY:
        raise LifecycleError("receipt sidecar type/mode")
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if side.read_text() != f"{digest}  {path.name}\n":
        raise LifecycleError("receipt sidecar")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LifecycleError("receipt JSON") from exc
    if not isinstance(value, dict):
        raise LifecycleError("receipt object")
    return value, digest


def _publish(receipt: Any, path: Path, payload: dict) -> tuple[dict, str]:
    receipt.write_receipt_transactionally(path, payload)
    return _descriptor_payload(path)


def _leaf_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise LifecycleError("non-file artifact leaf")
        name = path.name[:-7] if path.name.endswith(".sha256") else path.name
        names.add(name)
    return names


def _validate_bundle(bundle: Path, witness: BundleWitness | None = None) -> None:
    if witness is not None:
        bundle = witness.verify(bundle.parent)
    if bundle.is_symlink() or not bundle.is_dir() or _leaf_names(bundle) != _BUNDLE_NAMES:
        raise LifecycleError("committed score bundle topology")
    for name in _BUNDLE_NAMES:
        _descriptor_payload(bundle / name)


def _validate_result_topology(root: Path, *, failure: bool) -> None:
    names = set(path.name for path in root.iterdir())
    if failure:
        expected = {name for body in _FAILURE_NAMES for name in (body, body + ".sha256")}
        if names != expected:
            raise LifecycleError("failure root topology")
        for name in _FAILURE_NAMES:
            _descriptor_payload(root / name)
        return
    expected = {"attempt.json", "attempt.json.sha256", "complete"}
    if names != expected:
        raise LifecycleError("success root topology")
    _descriptor_payload(root / "attempt.json")
    bundle = root / "complete"
    _validate_bundle(bundle)
    terminal, _terminal_sha = _descriptor_payload(bundle / "terminal.json")
    claimed = terminal.get("committed_bundle")
    if not isinstance(claimed, dict) or claimed != BundleWitness.staged(root, bundle).__dict__:
        raise LifecycleError("terminal committed bundle binding")


def _remove_staging(stage: Path) -> None:
    """Staging is private/noncanonical and is the only mutable artifact."""
    if not stage.exists() and not stage.is_symlink():
        return
    if stage.is_symlink() or not stage.is_dir() or not stage.name.startswith(".stage-"):
        raise LifecycleError("staging identity")
    shutil.rmtree(stage)


def _fixed_session_bootstrap(values: list[float], seed: int) -> dict:
    """Small deterministic bootstrap primitive; rows, not callback summaries, drive it."""
    if not values:
        raise LifecycleError("empty bootstrap")
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(2_000))
    return {"seed": seed, "replicates": 2_000, "ci95": [means[49], means[1949]]}


def _validate_score_inputs(rows: list[Mapping[str, Any]], authority: Mapping[str, Any]) -> None:
    """Validate all callback material before it can become a canonical leaf."""
    from . import score
    if set(authority) != {"rosters", "records", "records_sha256"}:
        raise LifecycleError("input authority schema")
    rosters = authority["rosters"]
    records = authority["records"]
    if not isinstance(rosters, Mapping) or set(rosters) != set(plan.SURFACE_ORDER):
        raise LifecycleError("input authority rosters")
    normalized_rosters = {surface: tuple(rosters[surface]) for surface in plan.SURFACE_ORDER}
    if any(len(normalized_rosters[surface]) != plan.ROSTER_COUNTS[surface] or len(set(normalized_rosters[surface])) != len(normalized_rosters[surface]) for surface in plan.SURFACE_ORDER):
        raise LifecycleError("input authority roster cardinality")
    if not isinstance(records, list) or len(records) != 126 or _digest(records) != authority["records_sha256"]:
        raise LifecycleError("input authority records digest")
    expected_record_keys = set(score.InputRecord.__dataclass_fields__)
    if any(not isinstance(record, Mapping) or set(record) != expected_record_keys for record in records):
        raise LifecycleError("input authority record schema")
    canonical_rows = score.canonicalize_rows(rows, normalized_rosters)
    record_by_key = {(r["surface"], r["session"], int(r["budget"]), r["regime"]): dict(r) for r in records}
    if len(record_by_key) != 126:
        raise LifecycleError("input authority record topology")
    for row in canonical_rows:
        key = (row["surface"], row["session"], int(row["budget"]), row["regime"])
        if {field: row[field] for field in expected_record_keys} != record_by_key.get(key):
            raise LifecycleError("row/input-authority drift")


def _canonical_profile(profile: Mapping[str, Any]) -> dict:
    """The score route is CPU-only, so it cannot interfere with GPU jobs."""
    required = {"cuda_visible_devices", "selected_device", "cuda_initialized"}
    if set(profile) != required:
        raise LifecycleError("runtime/device profile schema")
    if profile["cuda_visible_devices"] != "" or profile["selected_device"] != "cpu" or profile["cuda_initialized"] is not False:
        raise LifecycleError("runtime/device profile")
    return dict(profile)


def _live_cpu_attestation() -> dict:
    """Route-owned live attestation; no caller-provided device claims."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise LifecycleError("CPU scorer requires empty CUDA_VISIBLE_DEVICES")
    import torch
    return _canonical_profile({"cuda_visible_devices": "", "selected_device": "cpu",
                               "cuda_initialized": bool(torch.cuda.is_initialized())})


def prepare_live_authority(binding: Any, *, repository_root: Path, receipt: Any,
                           score_root_relative: str | None = None,
                           authority_root_relative: str | None = None,
                           execution_profile: ScoreExecutionProfile = V1_SCORE_PROFILE) -> ScoreCapability:
    """Target-free preflight and one-shot capability issuance.

    It intentionally cannot run today because `PACDProducerBinding.require_live`
    rejects absent reviewed producer literals before any target-facing action.
    """
    _require_binding_type(binding, execution_profile)
    binding.require_live()
    score_root_relative = execution_profile.score_root_relative if score_root_relative is None else score_root_relative
    authority_root_relative = execution_profile.authority_root_relative if authority_root_relative is None else authority_root_relative
    if score_root_relative != execution_profile.score_root_relative or authority_root_relative != execution_profile.authority_root_relative:
        raise LifecycleError("canonical scorer roots required")
    profile = _live_cpu_attestation()
    score_root = RootWitness.fresh(repository_root, score_root_relative)
    authority_root = RootWitness.fresh(repository_root, authority_root_relative)
    if score_root.relative == authority_root.relative:
        raise LifecycleError("authority/result root alias")
    comparators = verify_historical_comparators(repository_root)
    producers = execution_profile.producer_validator(repository_root, binding)
    closure = source_closure(repository_root, execution_profile)
    # These receipts are target-free and are built in a private directory.
    # `authorized/` is published by one atomic directory rename only after
    # both descriptor bodies/sidecars reload successfully.
    authority_path = authority_root.verify_parent(repository_root)
    authority_path.mkdir(mode=0o755)
    authority_root = authority_root.capture_reserved(repository_root)
    stage = authority_path / ".stage-authority"
    stage.mkdir(mode=0o700)
    try:
        _preflight, preflight_sha = _publish(receipt, stage / "official_preflight.json", {
            "schema": execution_profile.schema + "_official_preflight", "status": "TARGET_FREE_PREFLIGHT",
            "binding_sha256": binding_sha256(binding), "producer_witness_sha256": _digest(producers),
            "comparator_witness_sha256": _digest(comparators), "closure_sha256": closure["closure_sha256"],
            "execution_profile": execution_profile.identity, "binding_type_identity": execution_profile.binding_type_identity,
            "runtime_profile": profile, "score_root": score_root.__dict__, "authority_root": authority_root.__dict__,
            "target_access": False,
        })
        _authorization, authorization_sha = _publish(receipt, stage / "root_authorization.json", {
            "schema": execution_profile.schema + "_root_authorization", "status": "ROOTS_AUTHORIZED",
            "official_preflight_sha256": preflight_sha, "binding_sha256": binding_sha256(binding),
            "closure_sha256": closure["closure_sha256"], "score_root": score_root.__dict__,
            "authority_root": authority_root.__dict__, "runtime_profile": profile, "target_access": False,
            "execution_profile": execution_profile.identity, "binding_type_identity": execution_profile.binding_type_identity,
        })
        _validate_authority_bundle(stage, preflight_sha, authorization_sha)
        os.rename(stage, authority_path / "authorized")
        _validate_authority_topology(authority_path, preflight_sha, authorization_sha)
    except BaseException:
        _remove_staging(stage)
        raise
    return ScoreCapability(_SECRET, binding_digest=binding_sha256(binding), score_root=score_root,
                           authority_root=authority_root, closure_sha256=closure["closure_sha256"],
                           producer_witness_sha256=_digest(producers), comparator_witness_sha256=_digest(comparators),
                           authority_sha256=authorization_sha, runtime_profile=profile, runtime_attestor=_live_cpu_attestation,
                           mode="live", execution_profile=execution_profile)


def _validate_authority_topology(root: Path, preflight_sha: str, authorization_sha: str) -> None:
    if set(path.name for path in root.iterdir()) != {"authorized"}:
        raise LifecycleError("authority artifact topology")
    _validate_authority_bundle(root / "authorized", preflight_sha, authorization_sha)


def _validate_authority_bundle(bundle: Path, preflight_sha: str, authorization_sha: str) -> None:
    if bundle.is_symlink() or not bundle.is_dir() or _leaf_names(bundle) != {"official_preflight.json", "root_authorization.json"}:
        raise LifecycleError("authority bundle topology")
    preflight, actual_preflight = _descriptor_payload(bundle / "official_preflight.json")
    authorization, actual_authorization = _descriptor_payload(bundle / "root_authorization.json")
    if actual_preflight != preflight_sha or actual_authorization != authorization_sha or authorization.get("official_preflight_sha256") != preflight_sha:
        raise LifecycleError("authority receipt links")
    if preflight.get("target_access") is not False or authorization.get("target_access") is not False:
        raise LifecycleError("authority target fact")


def _mint_synthetic_capability(secret: object, binding: Any, *, repository_root: Path,
                               root_relative: str, runtime_attestor: Callable[[], Mapping[str, Any]] | None = None,
                               execution_profile: ScoreExecutionProfile = V1_SCORE_PROFILE) -> ScoreCapability:
    """Typed test-only issuance; it never accepts a live binding."""
    if secret is not _SECRET or binding.mode != "synthetic":
        raise LifecycleError("synthetic capability is test-only")
    _require_binding_type(binding, execution_profile)
    score_root = RootWitness.fresh(repository_root, root_relative)
    authority_root = RootWitness.fresh(repository_root, root_relative + "_authority")
    placeholder = "0" * 64
    profile = {"cuda_visible_devices": "synthetic", "selected_device": "cpu", "cuda_initialized": False}
    return ScoreCapability(_SECRET, binding_digest=binding_sha256(binding), score_root=score_root,
                           authority_root=authority_root, closure_sha256=placeholder,
                           producer_witness_sha256=placeholder, comparator_witness_sha256=placeholder,
                           authority_sha256=placeholder, runtime_profile=profile,
                           runtime_attestor=runtime_attestor or (lambda: dict(profile)), mode="synthetic",
                           execution_profile=execution_profile)


def _revalidate_capability(capability: ScoreCapability, binding: Any, *, repository_root: Path,
                           out_dir: Path, allow_synthetic: bool, reserved: bool, internal: bool = False,
                           execution_profile: ScoreExecutionProfile = V1_SCORE_PROFILE) -> Path:
    _require_binding_type(binding, execution_profile)
    if not isinstance(capability, ScoreCapability) or capability._binding_digest != binding_sha256(binding):
        raise LifecycleError("binding/capability drift")
    if (capability._consumed and not internal) or binding.mode != capability._mode:
        raise LifecycleError("consumed or mode-drift capability")
    if capability._profile_identity != execution_profile.identity or capability._binding_type_identity != execution_profile.binding_type_identity:
        raise LifecycleError("execution-profile drift")
    expected = capability._score_root.verify_reserved(repository_root) if reserved else capability._score_root.verify_parent(repository_root)
    if out_dir != expected:
        raise LifecycleError("canonical score root identity")
    if not allow_synthetic and (binding.mode != "live" or capability._score_root.relative != execution_profile.score_root_relative):
        raise LifecycleError("live binding/canonical root required")
    if dict(capability._runtime_attestor()) != capability._runtime_profile:
        raise LifecycleError("runtime/device profile drift")
    if binding.mode == "live":
        closure = source_closure(repository_root, execution_profile)["closure_sha256"]
        if closure != capability._closure_sha256:
            raise LifecycleError("closure drift")
        authority_root = capability._authority_root.verify_reserved(repository_root)
        _validate_authority_topology(authority_root,
                                     _descriptor_payload(authority_root / "authorized" / "official_preflight.json")[1],
                                     capability._authority_sha256)
        # Descriptor graph revalidation is deliberately repeated at every
        # target-facing boundary; no stale live capability remains useful.
        if _digest(execution_profile.producer_validator(repository_root, binding)) != capability._producer_witness_sha256 or _digest(verify_historical_comparators(repository_root)) != capability._comparator_witness_sha256:
            raise LifecycleError("producer/comparator witness drift")
    return expected


def execute_atomic(*, repository_root: Path, out_dir: Path, binding: Any, capability: ScoreCapability,
                   receipt: Any, materialize_then_score: Callable[[], Mapping[str, Any]], allow_synthetic: bool = False,
                   _synthetic_after_commit: Callable[[Path], None] | None = None,
                   execution_profile: ScoreExecutionProfile = V1_SCORE_PROFILE) -> dict:
    """Publish attempt before materialization, then immutable score XOR failure."""
    target = _revalidate_capability(capability, binding, repository_root=repository_root, out_dir=out_dir,
                                    allow_synthetic=allow_synthetic, reserved=False, execution_profile=execution_profile)
    if target.exists():
        raise LifecycleError("immutable result root is not fresh")
    target.mkdir(mode=0o755)
    capability._score_root = capability._score_root.capture_reserved(repository_root)
    capability._consumed = True
    attempt, attempt_sha = _publish(receipt, target / "attempt.json", {
        "schema": execution_profile.schema + "_attempt", "status": "ATTEMPT_PUBLISHED", "binding_sha256": capability._binding_digest,
        "closure_sha256": capability._closure_sha256, "authority_root_relative": capability._authority_root.relative,
        "authority_sha256": capability._authority_sha256, "runtime_profile": capability._runtime_profile,
        "target_materialized": False, "checkpoint_deserialized": False, "cuda_initialized": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "execution_profile": execution_profile.identity, "binding_type_identity": execution_profile.binding_type_identity,
    })
    stage = target / f".stage-{attempt_sha[:16]}"
    committed = False
    bundle_witness: BundleWitness | None = None
    try:
        # Recheck after immutable attempt, immediately before target materialize.
        _revalidate_capability(capability, binding, repository_root=repository_root, out_dir=out_dir,
                               allow_synthetic=allow_synthetic, reserved=True, internal=True, execution_profile=execution_profile)
        callback_result = dict(materialize_then_score())
        input_authority = callback_result.pop("input_authority", None)
        if not isinstance(input_authority, Mapping):
            raise LifecycleError("materializer omitted input authority")
        rows = callback_result.get("rows")
        if not isinstance(rows, list) or len(rows) != execution_profile.expected_row_count:
            raise LifecycleError("incomplete score rows")
        _validate_score_inputs(rows, input_authority)
        # Recompute every aggregate, paired contrast, and gate from the exact
        # callback rows.  A caller cannot submit a favorable summary payload.
        from .score import assemble_score_result
        result = assemble_score_result(rows, bootstrap=_fixed_session_bootstrap)
        # This is the final live closure/graph/root revalidation.  Keep it
        # before any canonical score leaf is published so a drift can only
        # yield attempt→failure, never a partial score.
        _revalidate_capability(capability, binding, repository_root=repository_root, out_dir=out_dir,
                               allow_synthetic=allow_synthetic, reserved=True, internal=True, execution_profile=execution_profile)
        stage.mkdir(mode=0o700)
        bundle_witness = BundleWitness.staged(target, stage)
        authority, authority_sha = _publish(receipt, stage / "input_authority.json", {
            "schema": execution_profile.schema + "_input_authority", "attempt_sha256": attempt_sha,
            "target_materialized": True, "authority": dict(input_authority),
        })
        score, score_sha = _publish(receipt, stage / "score.json", {
            "schema": execution_profile.schema + "_score", "attempt_sha256": attempt_sha,
            "input_authority_sha256": authority_sha, "result": result,
        })
        terminal, _terminal_sha = _publish(receipt, stage / "terminal.json", {
            "schema": execution_profile.schema + "_terminal", "status": "SCORE_COMPLETE", "attempt_sha256": attempt_sha,
            "input_authority_sha256": authority_sha, "score_sha256": score_sha,
            "binding_sha256": capability._binding_digest, "closure_sha256": capability._closure_sha256,
            "runtime_profile": capability._runtime_profile, "cuda_initialized": False,
            "committed_bundle": bundle_witness.__dict__, "target_access": True,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "execution_profile": execution_profile.identity, "binding_type_identity": execution_profile.binding_type_identity,
        })
        _validate_bundle(stage)
        if terminal.get("score_sha256") != score_sha or authority.get("attempt_sha256") != attempt_sha or score.get("input_authority_sha256") != authority_sha:
            raise LifecycleError("terminal descriptor links")
        # One directory rename is the only canonical score publication.
        _revalidate_capability(capability, binding, repository_root=repository_root, out_dir=out_dir,
                               allow_synthetic=allow_synthetic, reserved=True, internal=True, execution_profile=execution_profile)
        os.rename(stage, target / "complete")
        committed = True
        assert bundle_witness is not None
        bundle_witness.verify(target)
        if _synthetic_after_commit is not None:
            if not allow_synthetic or binding.mode != "synthetic":
                raise LifecycleError("test-only commit hook")
            _synthetic_after_commit(target / "complete")
        _validate_result_topology(target, failure=False)
        return result
    except BaseException as error:
        if committed:
            # The immutable `complete/` directory was already atomically
            # published.  Never create a contradictory failure receipt.
            raise error
        # If the named root itself was swapped, never publish a failure into
        # the attacker replacement.  The retained original attempt root is
        # immutable evidence; caller receives the identity violation.
        try:
            capability._score_root.verify_reserved(repository_root)
        except LifecycleError:
            raise error
        final_revalidation = False
        try:
            _revalidate_capability(capability, binding, repository_root=repository_root, out_dir=out_dir,
                                   allow_synthetic=allow_synthetic, reserved=True, internal=True, execution_profile=execution_profile)
            final_revalidation = True
        except BaseException:
            # The original stage error remains the causal failure, while the
            # receipt makes closure/root revalidation status explicit.
            final_revalidation = False
        _remove_staging(stage)
        _publish(receipt, target / "failure.json", {
            "schema": execution_profile.schema + "_failure", "status": "SCORE_FAILED", "attempt_sha256": attempt_sha,
            "binding_sha256": capability._binding_digest, "closure_sha256": capability._closure_sha256,
            "failure": {"kind": type(error).__name__, "detail": str(error)},
            "progress": {"attempt_published": True, "canonical_input_authority_published": False,
                         "canonical_score_published": False, "canonical_terminal_published": False,
                         "staging_removed": True},
            "final_revalidation_passed": final_revalidation,
            "target_materialized": False, "checkpoint_deserialized": False,
            "runtime_profile": capability._runtime_profile, "cuda_initialized": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "execution_profile": execution_profile.identity, "binding_type_identity": execution_profile.binding_type_identity,
        })
        _validate_result_topology(target, failure=True)
        for name in ("attempt.json", "failure.json"):
            if (target / name).exists():
                _descriptor_payload(target / name)
        raise
