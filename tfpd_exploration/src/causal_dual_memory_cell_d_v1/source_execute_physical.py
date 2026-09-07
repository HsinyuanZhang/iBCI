"""Deferred physical strict-27 backend for CDM-D Source Execution V1.

Nothing in this module opens an NWB, checkpoint, or CUDA device at import.
Those actions are reachable only through ``source_execute.execute_authorized``
after a root-reviewed capability has passed the static lifecycle gate.  The
implementation deliberately composes the accepted Stage-0 and Source-Audit
types rather than reimplementing T4 fitting, B3S/native views, grouped B8, or
the Cell-D forward contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import source_execute as execute


class SourceExecutionPhysicalError(execute.SourceExecutionError):
    """Fail closed for the deferred physical source route."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionPhysicalError(message)


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be an exact lowercase SHA256")
    return value


@dataclass(frozen=True)
class FinalizedFourGroupPseudo:
    """One completed trial after all K=4 held-group outcomes are final.

    A true source direction is intentionally absent.  The backend calls the
    separate audit-only joiner only after this typed object is accepted, so a
    label cannot become a forward/state input through this interface.
    """

    session_id: str
    trial_id: str
    pseudo_direction_indices: tuple[int | None, ...]
    rejection_reasons: tuple[object | None, ...]
    prefix_digest: str
    physical_evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and self.session_id.startswith("sub-C_ses-CO-"),
                 "finalized pseudo session drift")
        _require(isinstance(self.trial_id, str) and self.trial_id, "finalized pseudo trial drift")
        _require(len(self.pseudo_direction_indices) == len(self.rejection_reasons) == execute.GROUP_COUNT,
                 "finalized pseudo must contain exactly four held-group outcomes")
        _require(isinstance(self.prefix_digest, str) and len(self.prefix_digest) == 64,
                 "finalized pseudo prefix digest drift")
        _require(isinstance(self.physical_evidence, Mapping)
                 and self.physical_evidence.get("all_four_groups_finalized") is True
                 and self.physical_evidence.get("group_label_broadcast") is False
                 and self.physical_evidence.get("all_group_inputs_physically_sliced") is True
                 and self.physical_evidence.get("held_group_count") == execute.GROUP_COUNT
                 and self.physical_evidence.get("label_join_before_outcomes") is False,
                 "finalized pseudo physical group evidence drift")


@dataclass(frozen=True)
class VariablePrefixHeldUnitInputs:
    """One physically held Cell-D input with its *actual* causal B3S prefix.

    The accepted generic Stage-0 physical seam deliberately exercises a full
    30-row B3S stack.  That is a useful compatibility check, but it is not the
    live causal contract for M4/M10: immediately after support sealing there
    are only ``M`` rows, and every accepted completed trial adds exactly one
    query row up to the immutable 30-row cap.  This route-local type makes the
    varying first axis explicit so a caller cannot silently zero-pad, repeat
    support, or preload a future trial merely to satisfy the old full-stack
    synthetic seam.
    """

    neural_windows: Any
    b3s_activity_stack: Any
    normalized_t4: Any
    retained_channel_indices: Any
    held_channel_indices: Any
    expected_prefix_length: int
    prefix_activity_sha256: str

    def __post_init__(self) -> None:
        # NumPy is intentionally deferred: importing this module must remain
        # source/checkpoint/CUDA free, while a real forward is already gated.
        import numpy as np

        neural = np.asarray(self.neural_windows)
        stack = np.asarray(self.b3s_activity_stack)
        side = np.asarray(self.normalized_t4)
        retained = np.asarray(self.retained_channel_indices)
        held = np.asarray(self.held_channel_indices)
        _require(neural.ndim == 3 and neural.shape[1] == 50 and neural.shape[2] >= 1,
                 "variable-prefix held neural input must be [P,50,N_remaining]")
        _require(stack.ndim == 3 and stack.shape[1] == 100 and stack.shape[2] == neural.shape[2],
                 "variable-prefix held B3S stack must be [M,100,N_remaining]")
        _require(1 <= stack.shape[0] <= 30 and stack.shape[0] == self.expected_prefix_length,
                 "variable-prefix B3S stack length drift")
        _require(side.shape == (neural.shape[2], 4),
                 "variable-prefix held normalized T4 shape drift")
        _require(retained.ndim == held.ndim == 1 and retained.size == neural.shape[2] and held.size >= 1,
                 "variable-prefix held channel-index shape drift")
        _require(np.issubdtype(retained.dtype, np.integer) and np.issubdtype(held.dtype, np.integer)
                 and np.isfinite(neural).all() and np.isfinite(stack).all() and np.isfinite(side).all(),
                 "variable-prefix held inputs must be finite and index-stable")
        _require_sha(self.prefix_activity_sha256, "variable-prefix activity SHA")
        # Values are copied by the caller's NumPy/Torch conversion boundary;
        # this type is only a local shape/provenance certificate.


def _variable_prefix_array_digest(value: Any) -> str:
    """Hash a finite contiguous NumPy view without importing shared seams."""
    import hashlib
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    _require(array.ndim == 3 and np.isfinite(array).all(),
             "variable-prefix activity digest needs finite [M,100,N] data")
    header = f"{array.dtype}|{tuple(int(item) for item in array.shape)}|".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def physically_slice_variable_prefix_held_units(
    neural_windows: Any,
    b3s_activity_stack: Any,
    normalized_t4: Any,
    held_unit_mask: Any,
    *,
    expected_prefix_length: int,
    full_prefix_activity_sha256: str | None = None,
) -> VariablePrefixHeldUnitInputs:
    """Physically slice a causal B3S prefix without fixed-30 substitution.

    ``expected_prefix_length`` comes from ``M + accepted_before_j`` on the
    owning :class:`core.ActivityMemory`, never from an arbitrary padded input.
    The optional full-prefix digest is calculated *before* held-unit slicing,
    binding this call to the exact support-plus-accepted history and rejecting
    a future-row injection even if the resulting tensor has the right shape.
    """
    import numpy as np

    neural = np.asarray(neural_windows)
    stack = np.asarray(b3s_activity_stack)
    side = np.asarray(normalized_t4)
    held = np.asarray(held_unit_mask)
    _require(neural.ndim == 3 and neural.shape[1] == 50,
             "variable-prefix neural windows must be [P,50,N]")
    _require(stack.ndim == 3 and stack.shape[1] == 100,
             "variable-prefix B3S stack must be [M,100,N]")
    _require(type(expected_prefix_length) is int and 1 <= expected_prefix_length <= 30
             and stack.shape[0] == expected_prefix_length,
             "variable-prefix stack must equal the exact causal support/query length")
    _require(side.ndim == 2 and side.shape[1] == 4
             and neural.shape[2] == stack.shape[2] == side.shape[0],
             "variable-prefix unit-aligned inputs drift")
    _require(held.dtype == np.bool_ and held.shape == (neural.shape[2],)
             and bool(held.any()) and bool((~held).any()),
             "variable-prefix held mask must remove a nonempty proper group")
    observed = _variable_prefix_array_digest(stack)
    if full_prefix_activity_sha256 is not None:
        _require_sha(full_prefix_activity_sha256, "expected variable-prefix activity SHA")
        _require(observed == full_prefix_activity_sha256,
                 "variable-prefix B3S input is not the exact causal support-plus-accepted stack")
    retained = np.flatnonzero(~held).astype(np.int64, copy=False)
    held_indices = np.flatnonzero(held).astype(np.int64, copy=False)
    return VariablePrefixHeldUnitInputs(
        neural[:, :, retained], stack[:, :, retained], side[retained], retained,
        held_indices, expected_prefix_length, observed,
    )


@dataclass(frozen=True)
class SourceThetaTopology:
    """Source-row order plus theta validity, without a reusable group map.

    The sealed theta authority is allowed to bind only these two facts.  Each
    live CDM-D budget must derive its own complementary held-unit map from its
    own support-fitted carrier; storing a raw-M30 group assignment here would
    make it far too easy to reuse that assignment at M4/M10.
    """

    channel_ids: Any = field(repr=False, compare=False)
    valid_mask: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        import numpy as np

        channels = np.asarray(self.channel_ids)
        valid = np.asarray(self.valid_mask)
        _require(channels.ndim == valid.ndim == 1 and channels.size >= execute.GROUP_COUNT,
                 "source theta topology must be nonempty aligned [N]")
        _require(np.issubdtype(channels.dtype, np.integer)
                 and len(set(int(value) for value in channels.tolist())) == channels.size,
                 "source theta topology channel IDs drift")
        _require(valid.dtype == np.bool_ and int(valid.sum()) >= execute.GROUP_COUNT,
                 "source theta topology valid-mask drift")
        object.__setattr__(self, "channel_ids", np.ascontiguousarray(channels, dtype=np.int64))
        object.__setattr__(self, "valid_mask", np.ascontiguousarray(valid, dtype=np.bool_))

    def payload(self) -> dict[str, object]:
        core, _source_adapter, _source_audit = _load_runtime_primitives()
        return {
            "channel_order_sha256": core.array_digest(self.channel_ids),
            "valid_mask_sha256": core.array_digest(self.valid_mask),
            "total_unit_count": int(self.channel_ids.size),
            "valid_unit_count": int(self.valid_mask.sum()),
            "invalid_unit_count": int((~self.valid_mask).sum()),
            "invalid_rows_unassigned_until_budget_specific_carrier": True,
        }


@dataclass(frozen=True)
class Strict27SessionMaterial:
    """Typed source-session material supplied only by a reviewed parser.

    ``trial_views_by_id`` contains the accepted independent B3S/native/validity
    capabilities.  The physical route validates their types only after the
    explicit execution path imports the accepted Source-Audit primitives.
    """

    session_id: str
    ordered_rewarded_trial_ids: tuple[str, ...]
    trial_views_by_id: Mapping[str, object]
    source_theta_topology: SourceThetaTopology
    m4_d_optimal_indices: tuple[int, ...]
    raw_t4_channel_order_sha256: str
    theta_valid_mask_sha256: str
    theta_invalid_unit_count: int
    source_descriptor_sha256: str
    # The following route-local fields are intentionally optional at the
    # protocol-only/synthetic boundary.  The concrete reviewed provider fills
    # every one before a physical Cell-D forward is reachable.
    raw_m30_t4: Any | None = field(default=None, repr=False, compare=False)
    neural_windows_by_id: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    support_direction_indices_by_id: Mapping[str, int] = field(default_factory=dict, repr=False, compare=False)
    # These rows intentionally use the historical fixed-ridge rate domain:
    # raw spike counts in the exact half-open rewarded interval divided by the
    # exact trial exposure in seconds.  They are distinct from the typed
    # native-20-ms counts used only by online completed-query updates.
    exact_duration_support_rates_by_id: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    exact_duration_support_exposure_seconds_by_id: Mapping[str, float] = field(
        default_factory=dict, repr=False, compare=False,
    )
    theta_recovery_evidence: Mapping[str, object] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and self.session_id.startswith("sub-C_ses-CO-"),
                 "strict27 material session drift")
        ids = tuple(self.ordered_rewarded_trial_ids)
        _require(len(ids) >= 30 and len(set(ids)) == len(ids)
                 and all(isinstance(item, str) and item for item in ids),
                 "strict27 rewarded chronology drift")
        _require(tuple(self.trial_views_by_id) == ids,
                 "strict27 trial-view mapping must retain chronology order exactly")
        _require(self.m4_d_optimal_indices == tuple(sorted(self.m4_d_optimal_indices))
                 and len(self.m4_d_optimal_indices) == 4
                 and len(set(self.m4_d_optimal_indices)) == 4
                 and all(type(item) is int and 0 <= item < 30 for item in self.m4_d_optimal_indices),
                 "strict27 sealed M4 D-opt support drift")
        for label, value in (
            ("raw T4 channel order", self.raw_t4_channel_order_sha256),
            ("theta valid-mask", self.theta_valid_mask_sha256),
            ("source descriptor", self.source_descriptor_sha256),
        ):
            _require(isinstance(value, str) and len(value) == 64, f"strict27 {label} SHA drift")
        _require(type(self.theta_invalid_unit_count) is int and self.theta_invalid_unit_count >= 0,
                 "strict27 theta invalid-unit count drift")
        _require(isinstance(self.source_theta_topology, SourceThetaTopology)
                 and self.source_theta_topology.channel_ids.size == self.source_theta_topology.valid_mask.size
                 and int((~self.source_theta_topology.valid_mask).sum()) == self.theta_invalid_unit_count,
                 "strict27 source theta validity topology drift")
        if self.neural_windows_by_id:
            _require(tuple(self.neural_windows_by_id) == ids,
                     "strict27 neural-window mapping must retain chronology order exactly")
        if self.support_direction_indices_by_id:
            _require(
                set(self.support_direction_indices_by_id).issubset(set(ids))
                and all(type(value) is int and 0 <= value < 8
                        for value in self.support_direction_indices_by_id.values()),
                "strict27 support-direction capability drift",
            )
        if self.exact_duration_support_rates_by_id or self.exact_duration_support_exposure_seconds_by_id:
            import numpy as np

            expected = set(ids[:30])
            rate_keys = set(self.exact_duration_support_rates_by_id)
            exposure_keys = set(self.exact_duration_support_exposure_seconds_by_id)
            _require(rate_keys == exposure_keys == expected,
                     "strict27 exact-duration support-rate topology drift")
            units = int(self.source_theta_topology.channel_ids.size)
            for trial_id in ids[:30]:
                rate = np.asarray(self.exact_duration_support_rates_by_id[trial_id], dtype=np.float64)
                exposure = self.exact_duration_support_exposure_seconds_by_id[trial_id]
                _require(rate.shape == (units,) and np.isfinite(rate).all()
                         and isinstance(exposure, (float, int)) and not isinstance(exposure, bool)
                         and np.isfinite(float(exposure)) and float(exposure) > 0.0,
                         "strict27 exact-duration support rate/exposure drift")
        object.__setattr__(self, "ordered_rewarded_trial_ids", ids)


class Strict27SessionProvider(Protocol):
    """Reviewed parser seam; called only after a durable attempt.

    ``manifest_roster`` is the immutable strict-27 authority.  ``open_roster``
    is intentionally narrower for smoke (exactly the fixed one session) and
    equals it only for the full source gate.  The split prevents a harmless
    manifest bind from turning into an unauthorized 27-file smoke open.
    """

    def prebind_train_only(
        self,
        *,
        manifest_roster: tuple[str, ...],
        open_roster: tuple[str, ...],
    ) -> Mapping[str, object]: ...
    def open_train_sessions(self, *, roster: tuple[str, ...]) -> Sequence[Strict27SessionMaterial]: ...
    def join_audit_true_direction(self, *, session_id: str, trial_id: str) -> object: ...
    def close(self) -> None: ...


class CellDFourGroupExecutor(Protocol):
    """Reviewed Cell-D eval/no-grad surface; no source labels are accepted."""

    def load_strict_sealed_swa(
        self, *, root: Path, swa_bytes: bytes, selected_device: Mapping[str, object], flags: execute.RuntimeFlags,
    ) -> Mapping[str, object]: ...
    def execute_completed_trial(
        self, *, material: Strict27SessionMaterial, budget: int, trial_id: str,
        support_trial_ids: tuple[str, ...], flags: execute.RuntimeFlags,
    ) -> FinalizedFourGroupPseudo: ...
    def begin_session_budget(
        self,
        *, material: Strict27SessionMaterial, budget: int,
        support_trial_ids: tuple[str, ...], flags: execute.RuntimeFlags,
    ) -> Mapping[str, object]: ...
    def groups_for_session_budget(
        self, *, material: Strict27SessionMaterial, budget: int,
    ) -> object: ...
    def resources(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


@dataclass
class _Runtime:
    assets: Mapping[str, execute.BoundAsset]
    manifest_roster: tuple[str, ...]
    physical_roster: tuple[str, ...]
    materials: Mapping[str, Strict27SessionMaterial]
    provider: Strict27SessionProvider
    executor: CellDFourGroupExecutor
    runtime_environment: Mapping[str, object]
    sealed_swa_load_proof: Mapping[str, object]
    selected_device: Mapping[str, object]
    prebind_evidence: Mapping[str, object]
    closed: bool = False


def _physical_roster_for_spec(identity: execute.SourceExecutionIdentity) -> tuple[str, ...]:
    """Return the sole source roster permitted to be opened by this run."""
    if identity.spec.kind == "source_smoke":
        _require(
            identity.spec.smoke_session == execute.SOURCE_SMOKE_SESSION
            and execute.SOURCE_SMOKE_SESSION in identity.strict_train_roster,
            "smoke strict-manifest/session binding drift",
        )
        return (execute.SOURCE_SMOKE_SESSION,)
    return tuple(identity.strict_train_roster)


def _validate_budget_memory_transition(
    evidence: Mapping[str, object],
    *,
    budget: int,
) -> None:
    """Bind the two memory domains separately after one finalized trial.

    M30's next-30 rows are a separate Source-Audit safety pool.  They may
    produce K=4 pseudo outcomes for offline grouped B8, but must not enter
    *either* deployment state: its zero-capacity activity FIFO and its carrier
    sufficient statistics remain exactly unchanged.  M4/M10, in contrast,
    exercise the normal causal dual-memory commit path.
    """
    for key in (
        "deployment_activity_memory_before_sha256",
        "deployment_activity_memory_after_sha256",
        "carrier_state_before_sha256",
        "carrier_state_after_sha256",
    ):
        _require_sha(evidence.get(key), f"physical memory transition {key}")
    _require(type(evidence.get("accepted_group_update_count")) is int
             and evidence["accepted_group_update_count"] in {0, execute.GROUP_COUNT},
             "physical memory transition accepted-group count drift")
    if budget != 30:
        return
    _require(
        evidence.get("activity_fifo_capacity") == 0
        and evidence.get("activity_query_count_before") == 0
        and evidence.get("activity_query_count_after") == 0
        and evidence.get("m30_audit_enters_deployment_activity_memory") is False
        and evidence["deployment_activity_memory_before_sha256"]
        == evidence["deployment_activity_memory_after_sha256"],
        "M30 zero-capacity activity/B3S memory transition drift",
    )
    _require(
        evidence["carrier_state_before_sha256"] == evidence["carrier_state_after_sha256"],
        "M30 safety-pool pseudo outcomes may not change carrier state",
    )


def _load_runtime_primitives() -> tuple[Any, Any, Any]:
    """Import accepted Stage-0/Source-Audit primitives only on execution path."""
    from . import core
    from . import source_adapter
    from . import source_audit
    return core, source_adapter, source_audit


def _validate_pmc_device_profile(profile: Mapping[str, object]) -> None:
    """Cross-bind dynamic selection to the one reviewed PMC profile source."""
    from src.posterior_marginalized_cell_d_v1 import plan as pmc_plan

    selected = execute.validate_compatible_device_profile(profile)
    _require(pmc_plan.validate_compatible_device_profile(selected) == selected,
             "CDM-D source execution device profile no longer matches reviewed PMC authority")


RuntimeClosureBuilder = Callable[[Path], Mapping[str, object]]


def _load_closure_bound_module(
    root: Path,
    *,
    relative: str,
    module_name: str,
    closure_builder: RuntimeClosureBuilder | None = None,
) -> Any:
    """Execute one direct runtime helper from descriptor-verified bytes.

    Importing ``src.tfpd_lane.pop_robust`` normally executes the lane package
    initializer, which in turn loads unrelated diagnostics.  This route needs
    only the two exact standalone helper files.  Reading their closure-bound
    bytes into private module namespaces preserves their real ``__file__``
    (the builder derives the repository root from it) without acquiring that
    ambient package-import closure.
    """
    import os
    import stat
    import sys
    import types

    # V1/V2 deliberately retain their original historical closure validator.
    # A successor may inject only an explicit callable that reconstructs its
    # own reviewed closure; there is no mutable-path/package-import fallback.
    builder: RuntimeClosureBuilder = execute.execution_closure_payload if closure_builder is None else closure_builder
    _require(callable(builder), "closure module builder must be an explicit callable")
    closure = builder(Path(root))
    rows = closure.get("paths") if isinstance(closure, Mapping) else None
    matches = [
        row.get("sha256") for row in rows
        if isinstance(row, Mapping) and row.get("path") == relative
    ] if isinstance(rows, list) else []
    _require(len(matches) == 1, f"closure-bound module {relative} row topology drift")
    expected = matches[0]
    _require_sha(expected, f"closure-bound module {relative}")
    path = Path(root).absolute() / relative
    parent = path.parent
    try:
        parent_lstat = os.lstat(parent)
        _require(stat.S_ISDIR(parent_lstat.st_mode) and not stat.S_ISLNK(parent_lstat.st_mode),
                 "closure module parent must be direct non-symlink")
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise SourceExecutionPhysicalError("closure module parent descriptor open failed") from error
    try:
        try:
            leaf_fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        except OSError as error:
            raise SourceExecutionPhysicalError("closure module leaf descriptor open failed") from error
        try:
            leaf = os.fstat(leaf_fd)
            _require(stat.S_ISREG(leaf.st_mode), "closure module must be regular non-symlink")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(leaf_fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            body = b"".join(chunks)
        finally:
            os.close(leaf_fd)
        after_named = os.lstat(parent)
        after_held = os.fstat(parent_fd)
        _require(
            (int(parent_lstat.st_dev), int(parent_lstat.st_ino))
            == (int(after_named.st_dev), int(after_named.st_ino))
            == (int(after_held.st_dev), int(after_held.st_ino)),
            "closure module parent identity changed during held read",
        )
    finally:
        os.close(parent_fd)
    _require(execute.sha256_bytes(body) == expected, "closure module bytes drift")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


def _runtime_imports(
    root: Path,
    *,
    closure_builder: RuntimeClosureBuilder | None = None,
) -> Mapping[str, Any]:
    """Load the closure-bound physical modules only after an attempt exists.

    This deliberately does not use a mutable package discovery helper.  The
    exact modules named here are all explicit members of
    ``execution_closure_payload``; adding one is therefore a launch-gate
    change, rather than an ambient-worktree dependency.
    """
    import importlib
    import sys

    base = Path(root).absolute()
    for path in (
        base / "tfpd_exploration",
        base / "sua_exploration",
        base / "streaming_calibration_exp",
    ):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    return {
        "torch": importlib.import_module("torch"),
        "numpy": importlib.import_module("numpy"),
        # CDM-D owns the typed B3S/native/velocity views.  The posterior-v2
        # route owns the audited raw-M30 T4 and same-prefix theta recovery.
        # They intentionally use different digest domains, so never collapse
        # them under one friendly-looking adapter name.
        "cdm_source_adapter": importlib.import_module("src.causal_dual_memory_cell_d_v1.source_adapter"),
        "posterior_core": importlib.import_module("src.posterior_carrier_v1.core"),
        "posterior_source_adapter": importlib.import_module("src.posterior_carrier_v1.source_adapter"),
        "posterior_source_adapter_v2": importlib.import_module("src.posterior_carrier_v1.source_adapter_v2"),
        "pop_robust": _load_closure_bound_module(
            base, relative="tfpd_exploration/src/tfpd_lane/pop_robust.py",
            module_name="cdmd_source_execution_pop_robust", closure_builder=closure_builder,
        ),
        "arm_common": _load_closure_bound_module(
            base, relative="tfpd_exploration/src/tfpd_lane/arm_common.py",
            module_name="cdmd_source_execution_arm_common", closure_builder=closure_builder,
        ),
        "multisession": importlib.import_module("mc_maze.multisession_datamodule"),
        "d_optimal": importlib.import_module("mc_maze.d_optimal_calibration_design"),
    }


def _file_sha256_from_fd(fd: int) -> str:
    """Stream one already-held source leaf without path re-resolution."""
    import hashlib
    import os

    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _descriptor_identity_sha256(*, device: int, inode: int, size: int, sha256: str) -> str:
    return execute.sha256_bytes(execute._json_bytes({
        "device": int(device), "inode": int(inode), "size": int(size), "sha256": _require_sha(sha256, "source leaf SHA"),
    }))


def _source_rng_snapshot(modules: Mapping[str, Any]) -> dict[str, str]:
    """Capture all four mutable RNG domains around a no-grad source forward."""
    import hashlib
    import pickle

    import random

    torch, np = modules["torch"], modules["numpy"]
    _require(torch.cuda.is_initialized(), "source forward RNG proof requires initialized selected CUDA")
    payload = {
        "python_sha256": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        "numpy_sha256": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        "torch_cpu_sha256": hashlib.sha256(bytes(torch.get_rng_state().cpu().numpy().tobytes())).hexdigest(),
        "torch_cuda_sha256": hashlib.sha256(bytes(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())).hexdigest(),
    }
    return payload


def _require_exact_rng_snapshot(value: object, *, label: str) -> dict[str, str]:
    _require(isinstance(value, Mapping) and set(value) == {
        "python_sha256", "numpy_sha256", "torch_cpu_sha256", "torch_cuda_sha256",
    }, f"{label} must bind Python/NumPy/Torch-CPU/Torch-CUDA RNG state")
    return {key: _require_sha(value[key], f"{label} {key}") for key in sorted(value)}


@dataclass(frozen=True)
class _ConcreteForwardTrial:
    """Route-private, source-only neural endpoint tensor for one trial."""

    neural_windows: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        import numpy as np

        value = np.asarray(self.neural_windows)
        _require(value.ndim == 3 and value.shape[1] == 50 and value.shape[0] >= 1
                 and value.shape[2] >= 4 and np.isfinite(value).all(),
                 "concrete trial neural endpoints must be finite [P,50,N>=4]")


def _rebuild_b3s_activity_from_held_native(
    held: Any,
    *,
    builder: Callable[..., Any],
) -> Any:
    """Rebuild one B3S activity view from the held native record alone.

    The accepted B3S bytes are an independently descriptor-bound reference;
    they are deliberately *not* an input to this callable.  Keeping the
    reconstruction in one route-local helper makes the concrete provider use
    exactly the same ``_build_calib_trials`` semantics as the direct parser,
    while preserving the source adapter's independent rebuild proof.
    """
    import numpy as np

    counts = np.asarray(held.full_native_binned_counts)
    channels = np.asarray(held.channel_ids)
    start = int(held.rewarded_interval_start_bin)
    stop = int(held.rewarded_interval_stop_bin)
    _require(counts.ndim == 2 and counts.shape[1] == channels.size
             and 0 <= start < stop <= counts.shape[0],
             "held-native B3S rebuilder input topology drift")
    rebuilt_batch = builder(
        counts,
        [{"start": start, "stop": stop}],
        1,
        100,
        int(channels.size),
        -1.0,
        True,
    )
    rebuilt = np.asarray(rebuilt_batch)
    _require(rebuilt.shape == (1, 100, channels.size),
             "held-native B3S rebuilder returned unexpected batch shape")
    result = np.ascontiguousarray(rebuilt[0])
    _require(np.issubdtype(result.dtype, np.floating) and np.isfinite(result).all(),
             "held-native B3S rebuild must be finite floating [100,N]")
    return result


def _exact_duration_support_rate_matrix(
    counts_by_unit_trial: Any,
    exposure_seconds_by_trial: Any,
) -> Any:
    """Return historical fixed-ridge support rates as ``counts.T / exposure``.

    This is intentionally separate from the Stage-0 online native-20-ms rate
    primitive.  The caller provides the already-approved raw spike counts over
    exact half-open trial intervals and their exact ``stop_time - start_time``
    durations, so no bin approximation can leak into the initial carrier.
    """
    import numpy as np

    counts = np.asarray(counts_by_unit_trial, dtype=np.float64)
    exposure = np.asarray(exposure_seconds_by_trial, dtype=np.float64)
    _require(counts.ndim == 2 and exposure.ndim == 1 and counts.shape[1] == exposure.size
             and counts.shape[0] >= execute.GROUP_COUNT
             and np.isfinite(counts).all() and np.all(counts >= 0.0)
             and np.isfinite(exposure).all() and np.all(exposure > 0.0),
             "exact-duration support count/exposure matrix drift")
    result = np.ascontiguousarray(counts.T / exposure[:, None], dtype=np.float64)
    _require(result.shape == (exposure.size, counts.shape[0]) and np.isfinite(result).all(),
             "exact-duration support rate matrix construction drift")
    result.setflags(write=False)
    return result


class ConcreteStrict27SessionProvider:
    """The reviewed no-cache, direct-child strict-source parser.

    It deliberately has no constructor side effects: the source directory is
    opened only from :meth:`open_train_sessions`, which the lifecycle calls
    after publishing attempt+launch.  The smoke gives it the complete manifest
    for authorization but the one-element physical roster for actual paths;
    full is the sole mode allowed to open all 27 direct children.
    """

    def __init__(self, *, root: Path, source_data: execute.StrictSourceDataRootCapability) -> None:
        self.root = Path(root).absolute()
        self.source_data = source_data
        self._manifest_roster: tuple[str, ...] | None = None
        self._open_roster: tuple[str, ...] | None = None
        self._directory_fd: int | None = None
        self._directory_identity: tuple[int, int] | None = None
        self._leaf_fds: list[int] = []
        self._audit_truth: dict[tuple[str, str], object] = {}
        self._closed = False

    def prebind_train_only(
        self,
        *,
        manifest_roster: tuple[str, ...],
        open_roster: tuple[str, ...],
    ) -> Mapping[str, object]:
        _require(not self._closed and self._manifest_roster is None,
                 "strict source provider may be prebound exactly once")
        _require(len(manifest_roster) == execute.STRICT_SOURCE_COUNT
                 and len(set(manifest_roster)) == execute.STRICT_SOURCE_COUNT,
                 "strict source provider manifest roster drift")
        _require(tuple(open_roster) and set(open_roster).issubset(set(manifest_roster)),
                 "strict source provider physical roster leaves strict train")
        # The source capability itself proves the direct root belongs to this
        # exact immutable roster.  No directory lstat, glob, path construction,
        # unit count, val/test resolver, or data opening occurs here.
        _require(
            self.source_data.strict_train_roster_sha256 == execute.roster_sha256(manifest_roster),
            "strict source provider capability/manifest roster drift",
        )
        self._manifest_roster = tuple(manifest_roster)
        self._open_roster = tuple(open_roster)
        return {
            "strict_train_roster": list(manifest_roster),
            "physical_open_roster": list(open_roster),
            "nontrain_path_or_unit_counter_called": False,
            "val_test_paths_constructed": False,
            "nonphysical_source_paths_resolved": False,
            "shared_datamodule_preclear_metadata_read_order": "not_used__route_direct_train_only_parser",
            "direct_child_filename_rule": "{session_id}_behavior+ecephys.nwb",
        }

    def _open_source_directory(self) -> int:
        import os
        import stat

        _require(self._manifest_roster is not None and self._open_roster is not None,
                 "strict source provider must be prebound before source open")
        if self._directory_fd is not None:
            return self._directory_fd
        try:
            fd = os.open(
                self.source_data.canonical_root,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            info = os.fstat(fd)
        except OSError as error:
            raise SourceExecutionPhysicalError("strict source root descriptor open failed") from error
        if not stat.S_ISDIR(info.st_mode):
            os.close(fd)
            raise SourceExecutionPhysicalError("strict source root is not a regular directory")
        self._directory_fd = fd
        self._directory_identity = (int(info.st_dev), int(info.st_ino))
        return fd

    def _open_leaf(self, *, session: str) -> tuple[int, Path, Mapping[str, object]]:
        import os
        import stat

        _require(self._manifest_roster is not None and self._open_roster is not None
                 and session in self._open_roster,
                 "strict source provider attempted a nonphysical source session")
        parent_fd = self._open_source_directory()
        filename = self.source_data.checked_relative_name(session, roster=self._manifest_roster)
        try:
            leaf_fd = os.open(filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            info = os.fstat(leaf_fd)
        except OSError as error:
            raise SourceExecutionPhysicalError(f"strict source leaf descriptor open failed: {session}") from error
        if not stat.S_ISREG(info.st_mode):
            os.close(leaf_fd)
            raise SourceExecutionPhysicalError("strict source leaf is not a regular non-symlink file")
        body_sha = _file_sha256_from_fd(leaf_fd)
        self._leaf_fds.append(leaf_fd)
        # The /proc fd path keeps the exact held file descriptor live across
        # each parser's re-open.  It is not a discovery pathname and has no
        # sibling/val/test resolution surface.
        parser_path = Path(f"/proc/self/fd/{leaf_fd}")
        return leaf_fd, parser_path, {
            "filename": filename,
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "size": int(info.st_size),
            "sha256": body_sha,
            "descriptor_sha256": _descriptor_identity_sha256(
                device=int(info.st_dev), inode=int(info.st_ino), size=int(info.st_size), sha256=body_sha,
            ),
        }

    @staticmethod
    def _theta_valid_mask(
        *, root: Path, session: str, raw_t4: Any, modules: Mapping[str, Any],
    ) -> tuple[Any, Mapping[str, object]]:
        """Descriptor-load the fixed theta artifact and bind raw/unit order."""
        torch, np = modules["torch"], modules["numpy"]
        from torch.nn.parameter import UninitializedParameter
        from torch.torch_version import TorchVersion

        bound = execute.descriptor_read_fixed_asset(
            root,
            next(asset for asset in execute.FIXED_ASSETS if asset.label == "theta_artifact"),
        )
        try:
            import io
            with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
                payload = torch.load(io.BytesIO(bound.body), map_location="cpu", weights_only=True)
        except Exception as error:
            raise SourceExecutionPhysicalError("theta authority weights-only load failed") from error
        _require(isinstance(payload, Mapping) and payload.get("kind") == "tfpd_sparsification_theta_authority_v1",
                 "theta authority payload schema drift")
        authority = payload.get("authority")
        _require(isinstance(authority, Mapping) and isinstance(authority.get(session), Mapping),
                 "theta authority strict source session row missing")
        row = authority[session]
        mask = row.get("valid")
        raw_sha = row.get("raw_t4_sha256")
        _require(torch.is_tensor(mask) and mask.ndim == 1 and mask.dtype == torch.bool,
                 "theta authority valid-mask dtype/shape drift")
        observed = np.asarray(raw_t4, dtype=np.float64)
        _require(int(row.get("n_units", -1)) == observed.shape[0]
                 and mask.numel() == observed.shape[0]
                 and isinstance(raw_sha, str) and len(raw_sha) == 64,
                 "theta authority raw/unit topology drift")
        # `unit_side_features` owns the source raw-T4 hash semantics.  We use
        # the closure-bound helper rather than inventing a raw-byte domain.
        posterior_core = modules["posterior_core"]
        cdm_source_adapter = modules["cdm_source_adapter"]
        actual_raw_sha = posterior_core.tensor_digest(torch.as_tensor(observed, dtype=torch.float64))
        _require(actual_raw_sha == raw_sha, "theta authority raw T4 bytes drift")
        expected_invalid = execute.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0)
        copied = mask.detach().cpu().numpy().astype(np.bool_, copy=True)
        _require(int((~copied).sum()) == expected_invalid,
                 "theta authority exact known undefined-unit count drift")
        return copied, {
            "theta_artifact_sha256": bound.asset.sha256,
            "raw_t4_sha256": raw_sha,
            "valid_mask_sha256": cdm_source_adapter.core.array_digest(copied),
            "invalid_unit_count": expected_invalid,
        }

    def _materialize_session(self, *, session: str, modules: Mapping[str, Any]) -> Strict27SessionMaterial:
        import math

        np, torch = modules["numpy"], modules["torch"]
        cdm_source_adapter = modules["cdm_source_adapter"]
        posterior_core = modules["posterior_core"]
        posterior_source_adapter = modules["posterior_source_adapter"]
        source_adapter_v2 = modules["posterior_source_adapter_v2"]
        multisession = modules["multisession"]
        d_optimal = modules["d_optimal"]
        core, _adapter, _audit = _load_runtime_primitives()
        _leaf_fd, parser_path, descriptor = self._open_leaf(session=session)
        # This is the same no-cache, source-only normalizer parameterization as
        # Cell-D.  No target/within/external array is ever constructed.
        record = multisession.load_dandi688_session(
            parser_path, bin_size_ms=20, window_size=50, calibration_n_trials=30,
            max_trial_length=100, pad_value=-1.0, interpolate_trials=True,
            behavior_mean=np.asarray(execute.SEALED_BEHAVIOR_MEAN, dtype=np.float32),
            behavior_std=np.asarray(execute.SEALED_BEHAVIOR_STD, dtype=np.float32),
            trial_result_filter="R", exclude_calibration_trials_from_windows=False,
            cache_dir=None, signal_view="sua",
        )
        _require(record.signal_view == "sua" and record.channel_ids is not None
                 and record.source_unit_count is not None,
                 "strict source direct parser signal/unit authority drift")
        units = int(record.source_unit_count)
        channels = np.asarray(record.channel_ids, dtype=np.int64)
        _require(record.neural.ndim == 2 and record.neural.shape[1] == units
                 and np.array_equal(channels, np.arange(units, dtype=np.int64)),
                 "strict source direct parser canonical channel order drift")
        trials = multisession.list_datamodule_rewarded_trials(
            parser_path, bin_size_ms=20, window_size=50, trial_result_filter="R",
        )
        _require(len(trials) >= 60, "strict source session lacks required 60 rewarded chronology rows")
        first60 = tuple(dict(item) for item in trials[:60])
        counts, exposure, theta, row_ids, recovery = source_adapter_v2._source_counts_exposure_theta_v2(
            path=parser_path, session=session, expected_units=units,
        )
        exact_count_matrix = np.asarray(counts.detach().cpu().numpy(), dtype=np.float64)
        exact_exposure_seconds = np.asarray(exposure.detach().cpu().numpy(), dtype=np.float64)
        _require(
            exact_count_matrix.shape == (units, 30)
            and exact_exposure_seconds.shape == (30,)
            and np.isfinite(exact_count_matrix).all()
            and np.all(exact_count_matrix >= 0.0)
            and np.isfinite(exact_exposure_seconds).all()
            and np.all(exact_exposure_seconds > 0.0),
            "strict source exact-duration support count/exposure topology drift",
        )
        exact_rate_matrix = _exact_duration_support_rate_matrix(
            exact_count_matrix, exact_exposure_seconds,
        )
        source_adapter_v2.validate_theta_recovery_evidence(
            recovery, session=session, prefix_row_ids=row_ids,
            theta_sha256=posterior_core.tensor_digest(theta),
        )
        raw_t4, unit_order_sha, raw_t4_proof = posterior_source_adapter._raw_m30_t4_and_unit_order(
            path=parser_path, session=session, expected_units=units,
            record_source_unit_count=units, record_channel_ids=channels,
        )
        raw = np.asarray(raw_t4.detach().cpu().numpy(), dtype=np.float64)
        valid_mask, theta_binding = self._theta_valid_mask(
            root=self.root, session=session, raw_t4=raw, modules=modules,
        )
        source_theta_topology = SourceThetaTopology(channels, valid_mask)
        _require(int((~source_theta_topology.valid_mask).sum())
                 == execute.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0),
                 "strict source theta topology lost known invalid rows")
        # The v2 recovered first-30 theta is the sole M4 D-opt authority;
        # later M30 safety rows must not substitute their labels for it.
        m4 = tuple(sorted(int(value) for value in d_optimal.greedy_forward_d_optimal_indices(
            np.asarray(theta.detach().cpu().numpy(), dtype=np.float64), 4,
        ).tolist()))
        ids: list[str] = []
        views: dict[str, object] = {}
        forward: dict[str, object] = {}
        support_directions: dict[str, int] = {}
        for position, item in enumerate(first60):
            original_index = int(item["trial_index"])
            trial_id = f"{session}:trial:{original_index}"
            ids.append(trial_id)
            start, stop = int(item["start"]), int(item["stop"])
            _require(0 <= start < stop <= int(record.neural.shape[0]) and stop - start >= 50,
                     "strict source rewarded trial interval/window drift")
            activity = multisession._build_calib_trials(
                record.neural, [{"start": start, "stop": stop}], 1, 100, units, -1.0, True,
            )[0]
            endpoints = np.arange(start + 49, stop, dtype=np.int64)
            windows = np.stack([record.neural[index - 49:index + 1] for index in endpoints], axis=0)
            descriptor_obj = cdm_source_adapter.SourceTrialDescriptor(
                session_id=session, trial_id=trial_id, source_record_sha256=descriptor["sha256"],
                channel_order_sha256=core.channel_order_digest(channels),
                full_native_counts_sha256=core.array_digest(record.neural),
                accepted_b3s_activity_sha256=core.array_digest(activity),
                raw_trial_binding_sha256=cdm_source_adapter.raw_trial_binding_sha256(
                    session_id=session, trial_id=trial_id, source_record_sha256=descriptor["sha256"],
                    channel_order_sha256=core.channel_order_digest(channels),
                    full_native_counts_sha256=core.array_digest(record.neural),
                    rewarded_interval_start_bin=start, rewarded_interval_stop_bin=stop,
                    accepted_b3s_activity_sha256=core.array_digest(activity),
                    window_endpoint_bins_sha256=core.array_digest(endpoints),
                    neural_endpoint_available_sha256=core.array_digest(np.ones(endpoints.shape, dtype=np.bool_)),
                    b3s_rebuilder_semantics=cdm_source_adapter.B3S_REBUILDER_SEMANTICS,
                ),
            )
            held = cdm_source_adapter.HeldSourceTrialRecord(
                descriptor=descriptor_obj, channel_ids=channels, full_native_binned_counts=record.neural,
                rewarded_interval_start_bin=start, rewarded_interval_stop_bin=stop,
                accepted_calibration_b3s_activity=activity,
                rebuild_b3s_activity=(
                    lambda held_record, builder=multisession._build_calib_trials:
                    _rebuild_b3s_activity_from_held_native(held_record, builder=builder)
                ),
                window_endpoint_bins=endpoints,
                neural_endpoint_available=np.ones(endpoints.shape, dtype=np.bool_),
            )
            view = cdm_source_adapter.materialize_source_trial_views(
                held, roster=cdm_source_adapter.StrictSourceRoster(tuple(self._manifest_roster or ())),
            )
            views[trial_id] = view
            forward[trial_id] = _ConcreteForwardTrial(np.asarray(windows, dtype=np.float32))
            if position < 30:
                support_directions[trial_id] = int(core.nearest_canonical_direction(
                    float(theta[position].item()),
                )[0])
            # Audit truth remains in the provider only and is not handed to
            # model/state code.  Same-prefix recovery is allowed only for the
            # already selected first-30 row; later labels must be native.
            direction = float(theta[position].item()) if position < 30 else item.get("target_dir")
            _require(direction is not None and math.isfinite(float(direction)),
                     "audit true direction missing outside approved same-prefix recovery")
            self._audit_truth[(session, trial_id)] = cdm_source_adapter.AuditOnlyTrueDirection(
                session, trial_id, float(direction),
            )
        _require(tuple(ids[:30]) == tuple(row_ids), "strict source v2 prefix trial identity drift")
        _require(raw_t4_proof.get("unit_order_sha256") == unit_order_sha,
                 "strict source raw-T4 unit-order proof drift")
        exact_duration_rates: dict[str, Any] = {}
        exact_duration_exposure: dict[str, float] = {}
        for position, trial_id in enumerate(ids[:30]):
            value = np.ascontiguousarray(exact_rate_matrix[position], dtype=np.float64)
            _require(value.shape == (units,) and np.isfinite(value).all(),
                     "strict source exact-duration support rate construction drift")
            value.setflags(write=False)
            exact_duration_rates[trial_id] = value
            exact_duration_exposure[trial_id] = float(exact_exposure_seconds[position])
        return Strict27SessionMaterial(
            session_id=session, ordered_rewarded_trial_ids=tuple(ids), trial_views_by_id=views,
            source_theta_topology=source_theta_topology, m4_d_optimal_indices=m4,
            raw_t4_channel_order_sha256=core.array_digest(channels),
            theta_valid_mask_sha256=theta_binding["valid_mask_sha256"],
            theta_invalid_unit_count=int(theta_binding["invalid_unit_count"]),
            source_descriptor_sha256=descriptor["descriptor_sha256"], raw_m30_t4=raw,
            neural_windows_by_id=forward, support_direction_indices_by_id=support_directions,
            exact_duration_support_rates_by_id=exact_duration_rates,
            exact_duration_support_exposure_seconds_by_id=exact_duration_exposure,
            theta_recovery_evidence=recovery,
        )

    def _runtime_modules(self) -> Mapping[str, Any]:
        """Historical V1/V2 runtime-module seam.

        Keeping this one-method indirection is intentionally backward
        compatible: subclasses can supply a reviewed successor closure only
        for the two descriptor-executed helpers, while the default path still
        invokes :func:`source_execute.execution_closure_payload` exactly.
        """
        return _runtime_imports(self.root)

    def open_train_sessions(self, *, roster: tuple[str, ...]) -> Sequence[Strict27SessionMaterial]:
        _require(not self._closed and self._open_roster is not None and tuple(roster) == self._open_roster,
                 "strict source provider open roster differs from prebound physical roster")
        modules = self._runtime_modules()
        result = tuple(self._materialize_session(session=session, modules=modules) for session in roster)
        _require(tuple(item.session_id for item in result) == tuple(roster),
                 "strict source provider session material ordering drift")
        return result

    def join_audit_true_direction(self, *, session_id: str, trial_id: str) -> object:
        key = (session_id, trial_id)
        _require(key in self._audit_truth, "audit-only true label requested before finalized physical outcome")
        return self._audit_truth[key]

    def close(self) -> None:
        import os

        if self._closed:
            return
        self._closed = True
        for fd in reversed(self._leaf_fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self._leaf_fds.clear()
        if self._directory_fd is not None:
            try:
                current = os.fstat(self._directory_fd)
                _require(self._directory_identity == (int(current.st_dev), int(current.st_ino)),
                         "strict source root descriptor identity drift before close")
            finally:
                os.close(self._directory_fd)
                self._directory_fd = None


@dataclass
class _ConcreteExecutorState:
    modules: Mapping[str, Any]
    model: Any
    device: Any
    behavior_normalizer: Any
    state_by_session_budget: dict[tuple[str, int], Any] = field(default_factory=dict)
    state_initial_evidence: dict[tuple[str, int], Mapping[str, object]] = field(default_factory=dict)
    forward_calls: int = 0
    endpoint_chunks_completed: int = 0
    completed_trials: int = 0
    measurement_started_monotonic: float = 0.0
    closed: bool = False


class ConcreteCellDFourGroupExecutor:
    """Strict sealed Cell-D executor with no global/model monkeypatches.

    The object owns only the adaptation from immutable CDM-D state snapshots to
    the unchanged coupled Cell-D forward.  It does not wrap the decoder, add a
    credibility bias, normalizer, table, or parameter, and it does not accept
    source true labels.  Every `(session, budget)` receives a fresh state.
    """

    def __init__(self, *, root: Path) -> None:
        self.root = Path(root).absolute()
        self._runtime: _ConcreteExecutorState | None = None
        # TF32 is process-global Torch state.  Preserve it exactly around this
        # route rather than leaving a successful or failed source gate able to
        # alter a later independent computation in the same interpreter.
        self._tf32_restore: tuple[Any, bool, bool] | None = None

    @staticmethod
    def _lazy_topology(model: Any, torch: Any) -> tuple[int, tuple[str, ...]]:
        from torch.nn.parameter import UninitializedParameter

        initialized = 0
        lazy: list[str] = []
        for name, parameter in model.named_parameters():
            if isinstance(parameter, UninitializedParameter):
                lazy.append(name)
            else:
                initialized += int(parameter.numel())
        return initialized, tuple(sorted(lazy))

    @staticmethod
    def _safe_swa_state(torch: Any, body: bytes) -> Mapping[str, object]:
        import io
        from torch.nn.parameter import UninitializedParameter
        from torch.torch_version import TorchVersion

        try:
            with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
                payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        except Exception as error:
            raise SourceExecutionPhysicalError("sealed Cell-D SWA weights-only load failed") from error
        _require(isinstance(payload, Mapping), "sealed Cell-D SWA payload root drift")
        state = payload.get("state_dict", payload.get("state"))
        manifest = payload.get("swa_manifest")
        _require(isinstance(state, Mapping) and isinstance(manifest, Mapping)
                 and manifest.get("uninitialized_lazy_tensor_count") == 2
                 and manifest.get("optimizer_state_included") is False
                 and manifest.get("fp64_arithmetic") is True
                 and isinstance(manifest.get("components"), list)
                 and len(manifest["components"]) == 4,
                 "sealed Cell-D SWA state/manifest topology drift")
        return state

    def _enforce_route_local_tf32_false(self, torch: Any) -> None:
        """Snapshot and enforce the reviewed TF32 state for this route only."""
        _require(self._tf32_restore is None,
                 "strict Cell-D source executor TF32 state is already owned by a live route")
        before_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
        before_cudnn = bool(torch.backends.cudnn.allow_tf32)
        self._tf32_restore = (torch, before_matmul, before_cudnn)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            _require(torch.backends.cuda.matmul.allow_tf32 is False
                     and torch.backends.cudnn.allow_tf32 is False,
                     "strict Cell-D source executor must enforce TF32 false before model forward")
        except BaseException:
            self._restore_route_local_tf32()
            raise

    def _restore_route_local_tf32(self) -> None:
        """Restore the exact process-global TF32 flags if this route changed them."""
        owned = self._tf32_restore
        self._tf32_restore = None
        if owned is None:
            return
        torch, before_matmul, before_cudnn = owned
        torch.backends.cuda.matmul.allow_tf32 = before_matmul
        torch.backends.cudnn.allow_tf32 = before_cudnn
        _require(bool(torch.backends.cuda.matmul.allow_tf32) == before_matmul
                 and bool(torch.backends.cudnn.allow_tf32) == before_cudnn,
                 "strict Cell-D source executor failed to restore route-local TF32 state")

    def _attest(self, torch: Any, profile: Mapping[str, object], *, flags: execute.RuntimeFlags) -> dict[str, object]:
        import subprocess

        selected = execute.validate_compatible_device_profile(profile)
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                 "strict Cell-D source executor requires exactly one visible CUDA device")
        props = torch.cuda.get_device_properties(0)
        flags.cuda_initialized = True
        _require(str(torch.__version__) == selected["torch_version"]
                 and str(torch.version.cuda) == selected["torch_cuda_version"]
                 and int(torch.backends.cudnn.version()) == selected["cudnn_version"]
                 and str(props.name) == selected["name"]
                 and int(props.total_memory) == selected["torch_total_memory_bytes"],
                 "strict Cell-D source executor Torch runtime authority drift")
        try:
            rows = subprocess.check_output(
                ["nvidia-smi", "-i", str(selected["cuda_visible_devices"]),
                 "--query-gpu=uuid,pci.bus_id,name,memory.total", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SourceExecutionPhysicalError("strict Cell-D source executor nvidia-smi attestation failed") from error
        _require(len(rows) == 1, "strict Cell-D source executor nvidia-smi row count drift")
        uuid, bdf, name, memory = (item.strip() for item in rows[0].split(",", 3))
        _require({"uuid": uuid, "bdf": bdf, "name": name, "nvidia_smi_memory_total_mib": int(memory)}
                 == {key: selected[key] for key in ("uuid", "bdf", "name", "nvidia_smi_memory_total_mib")},
                 "strict Cell-D source executor nvidia-smi authority drift")
        self._enforce_route_local_tf32_false(torch)
        return {
            **selected, "visible_devices": 1, "attested": True,
            "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False,
        }

    def _runtime_modules(self, root: Path) -> Mapping[str, Any]:
        """Historical V1/V2 SWA-loader runtime-module seam.

        This mirrors :meth:`ConcreteStrict27SessionProvider._runtime_modules`
        so a successor must opt in explicitly for both physical call sites.
        """
        return _runtime_imports(Path(root))

    def load_strict_sealed_swa(
        self, *, root: Path, swa_bytes: bytes, selected_device: Mapping[str, object], flags: execute.RuntimeFlags,
    ) -> Mapping[str, object]:
        _require(self._runtime is None, "strict Cell-D source executor may load sealed SWA once")
        modules = self._runtime_modules(Path(root))
        torch, np = modules["torch"], modules["numpy"]
        state = self._safe_swa_state(torch, swa_bytes)
        flags.checkpoint_opened = True
        runtime_environment = self._attest(torch, selected_device, flags=flags)
        device = torch.device("cuda:0")
        model = modules["pop_robust"].build_population_robustness_model(seed=42, cell="D")
        before_count, before_lazy = self._lazy_topology(model, torch)
        _require(before_count == execute.SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS
                 and before_lazy == execute.SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS
                 and set(state) == set(model.state_dict()),
                 "fresh Cell-D/SWA lazy state-key topology drift")
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        count, lazy = self._lazy_topology(model, torch)
        state_sha = modules["arm_common"].state_sha256(model)
        _require(count == execute.SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS
                 and lazy == execute.SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS
                 and isinstance(state_sha, str) and len(state_sha) == 64,
                 "strict Cell-D/SWA reload topology or digest drift")
        # A real B3S variable-M forward serves as the compatibility proof.  It
        # is not a scientific score and it consumes no source data.
        neural = torch.zeros((1, 50, 4), dtype=torch.float32, device=device)
        calib = torch.zeros((1, 1, 100, 4), dtype=torch.float32, device=device)
        side = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
        before = modules["arm_common"].state_sha256(model)
        with torch.no_grad():
            first, _ = model(neural, calib_trials=calib, side_features=side)
            second, _ = model(neural, calib_trials=calib, side_features=side)
        _require(bool(torch.isfinite(first).all().item()) and torch.equal(first, second)
                 and before == modules["arm_common"].state_sha256(model),
                 "strict Cell-D variable-prefix eval purity proof drift")
        source_physical = __import__("src.causal_dual_memory_cell_d_v1.physical", fromlist=["SealedSourceBehaviorNormalizer"])
        behavior = source_physical.SealedSourceBehaviorNormalizer(
            mean=np.asarray(execute.SEALED_BEHAVIOR_MEAN, dtype=np.float64),
            std=np.asarray(execute.SEALED_BEHAVIOR_STD, dtype=np.float64),
            authority_sha256=execute.SEALED_BEHAVIOR_NORMALIZER_SHA256,
        )
        # This is the reviewed measurement boundary: all strict-load and
        # fixed compatibility allocations precede it, while every source
        # held-group forward follows it.  The final receipt therefore reports
        # source-route peak memory rather than a stale process-global peak.
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        self._runtime = _ConcreteExecutorState(
            modules=modules, model=model, device=device, behavior_normalizer=behavior,
        )
        return {
            "runtime_environment": runtime_environment,
            "sealed_swa_load_proof": {
                "schema": execute.SEALED_SWA_LOAD_PROOF_SCHEMA,
                "sealed_terminal_sha256": execute.SEALED_CELL_D_TERMINAL_SHA256,
                "sealed_swa_sha256": execute.SEALED_CELL_D_SWA_SHA256,
                "fresh_strict_load": True,
                "recomputed_state_dict_sha256": state_sha,
                "initialized_trainable_parameters": count,
                "uninitialized_lazy_keys": list(lazy),
                "model_eval": True, "no_grad": True, "finite_forward": True,
                "repeated_fixed_forward_bitwise_equal": True, "model_state_unchanged": True,
                "dynamic_dropout_calls": 0,
            },
        }

    def _require_runtime(self) -> _ConcreteExecutorState:
        _require(self._runtime is not None and not self._runtime.closed,
                 "strict Cell-D source executor runtime is unavailable")
        return self._runtime

    @staticmethod
    def _support_rates_and_labels(
        *, material: Strict27SessionMaterial, support_trial_ids: tuple[str, ...], core: Any,
    ) -> tuple[Any, Any, Mapping[str, object]]:
        import numpy as np

        _require(material.support_direction_indices_by_id,
                 "concrete executor requires parser-bound sealed support labels")
        _require(material.exact_duration_support_rates_by_id
                 and material.exact_duration_support_exposure_seconds_by_id,
                 "concrete executor requires exact-duration fixed-ridge support rates")
        rates = []
        labels = []
        exposures = []
        for trial_id in support_trial_ids:
            view = material.trial_views_by_id.get(trial_id)
            _require(view is not None and trial_id in material.support_direction_indices_by_id,
                     "sealed support trial lacks typed native-rate/label capability")
            _require(
                trial_id in material.exact_duration_support_rates_by_id
                and trial_id in material.exact_duration_support_exposure_seconds_by_id,
                "sealed support trial lacks exact-duration comparator-rate capability",
            )
            # This is deliberately *not* scalar_rates_from_native_rewarded_counts:
            # the initial M4/M10/M30 fixed-ridge carrier must reproduce the
            # established half-open raw-spike-count/exact-exposure comparator.
            rate = np.asarray(material.exact_duration_support_rates_by_id[trial_id], dtype=np.float64)
            exposure = float(material.exact_duration_support_exposure_seconds_by_id[trial_id])
            _require(rate.shape == (material.source_theta_topology.channel_ids.size,)
                     and np.isfinite(rate).all() and np.isfinite(exposure) and exposure > 0.0,
                     "sealed support exact-duration rate/exposure drift")
            rates.append(rate)
            exposures.append(exposure)
            labels.append(int(material.support_direction_indices_by_id[trial_id]))
        rate_matrix = np.ascontiguousarray(np.stack(rates, axis=0), dtype=np.float64)
        exposure_array = np.ascontiguousarray(np.asarray(exposures, dtype=np.float64))
        _require(rate_matrix.shape == (len(support_trial_ids), material.source_theta_topology.channel_ids.size)
                 and exposure_array.shape == (len(support_trial_ids),)
                 and np.isfinite(rate_matrix).all() and np.all(exposure_array > 0.0),
                 "sealed support exact-duration matrix topology drift")
        return rate_matrix, np.asarray(labels, dtype=np.int64), {
            "initial_support_rate_domain": execute.INITIAL_SUPPORT_RATE_DOMAIN,
            "online_update_rate_domain": execute.ONLINE_UPDATE_RATE_DOMAIN,
            "initial_support_rates_sha256": core.array_digest(rate_matrix),
            "initial_support_exposure_seconds_sha256": core.array_digest(exposure_array),
        }

    @staticmethod
    def _budget_initial_carrier(
        *, rates: Any, labels: Any, budget: int, core: Any,
    ) -> tuple[Any, Mapping[str, object]]:
        """Fit the actual budget-limited deployment carrier before grouping.

        The source-wide raw M30 T4 authority proves raw feature/order and the
        theta-validity mask only.  It is expressly *not* an initializer for
        M4/M10: otherwise post-budget label information could leak into the
        first held prediction.  The accepted CDM-D source protocol selects
        the fixed-ridge-by-trial recipe at all three budgets, including the
        sealed chronological M30 support arm.
        """
        _require(budget in (4, 10, 30), "budget initial carrier budget drift")
        _require(rates.shape[0] == labels.size == budget,
                 "budget initial carrier support-row topology drift")
        mode = core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL
        candidate = core.fit_carriers_from_trial_table(
            rates, labels, mode=mode, normalized_lambda=core.RIDGE_NORMALIZED_LAMBDA,
        )
        parity = core.assert_production_parity(
            candidate, rates, labels, mode=mode,
        )
        return candidate, {
            "initial_carrier_recipe": mode.value,
            "initial_carrier_support_rows": int(budget),
            "initial_carrier_parity": parity,
            "initial_carrier_sha256": core.array_digest(candidate),
            "raw_m30_t4_used_as_initializer": False,
        }

    def begin_session_budget(
        self, *, material: Strict27SessionMaterial, budget: int,
        support_trial_ids: tuple[str, ...], flags: execute.RuntimeFlags,
    ) -> Mapping[str, object]:
        runtime = self._require_runtime()
        core, _adapter, _audit = _load_runtime_primitives()
        import numpy as np

        _require(budget in (4, 10, 30) and len(support_trial_ids) == budget,
                 "concrete executor sealed support budget topology drift")
        key = (material.session_id, budget)
        _require(key not in runtime.state_by_session_budget,
                 "concrete executor may not inherit/reuse a prior session/budget state")
        rates, labels, rate_evidence = self._support_rates_and_labels(
            material=material, support_trial_ids=support_trial_ids, core=core,
        )
        initial_t4, initial_evidence = self._budget_initial_carrier(
            rates=rates, labels=labels, budget=budget, core=core,
        )
        config = core.CDMDConfig(
            support_budget_m=budget,
            active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        carrier = core.CarrierMemory.from_support_trials(
            initial_raw_t4=initial_t4,
            channel_ids=material.source_theta_topology.channel_ids, support_trial_rates=rates,
            support_direction_indices=labels, config=config,
            valid_mask=material.source_theta_topology.valid_mask,
        )
        support_activity = tuple(material.trial_views_by_id[item].b3s_activity for item in support_trial_ids)
        activity = core.ActivityMemory.initialize(
            support_activity, channel_ids=material.source_theta_topology.channel_ids,
            fifo_capacity=int(config.activity_fifo_capacity),
        )
        memory = core.CausalDualMemory(activity=activity, carrier=carrier)
        runtime.state_by_session_budget[key] = memory
        evidence = {
            "session_id": material.session_id, "budget": budget,
            "support_trial_ids": list(support_trial_ids), "fresh_for_session_budget": True,
            "inherits_prior_session_or_budget_state": False,
            "initial_activity_memory_sha256": memory.state.activity.digest,
            "initial_carrier_state_sha256": memory.state.carrier.digest,
            "activity_fifo_capacity": int(config.activity_fifo_capacity),
            "initial_activity_query_count": int(memory.state.activity.query_count),
            "support_activity_trial_count": len(support_activity),
            "support_carrier_rate_rows": int(rates.shape[0]),
            **initial_evidence,
            **rate_evidence,
            "budget_groups_sha256": carrier.groups.digest,
            "budget_group_assignment_sha256": core.array_digest(carrier.groups.assignment),
            "budget_group_valid_mask_sha256": core.array_digest(carrier.groups.valid_mask),
        }
        runtime.state_initial_evidence[key] = evidence
        return evidence

    def groups_for_session_budget(
        self, *, material: Strict27SessionMaterial, budget: int,
    ) -> object:
        """Return the sole group map used by this budget's live consumer."""
        runtime = self._require_runtime()
        key = (material.session_id, budget)
        _require(key in runtime.state_by_session_budget,
                 "budget group map requested before fresh carrier initialization")
        return runtime.state_by_session_budget[key].state.carrier.groups

    @staticmethod
    def _normalized_active_t4(runtime: _ConcreteExecutorState, raw_t4: Any) -> Any:
        torch = runtime.modules["torch"]
        value = torch.as_tensor(raw_t4, dtype=torch.float32, device=runtime.device)
        mean = torch.as_tensor(execute.SEALED_T4_MEAN_FLOAT32, dtype=torch.float32, device=runtime.device)
        std = torch.as_tensor(execute.SEALED_T4_STD_FLOAT32, dtype=torch.float32, device=runtime.device)
        result = (value - mean) / std
        _require(result.ndim == 2 and result.shape[1] == 4 and bool(torch.isfinite(result).all().item()),
                 "concrete executor sealed OLS normalized T4 drift")
        return result

    @staticmethod
    def _torch_variable_prefix_forward(
        runtime: _ConcreteExecutorState,
        *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
        held_mask: Any, validity: Any, expected_prefix_length: int,
        expected_prefix_activity_sha256: str,
    ) -> tuple[Any, Mapping[str, object]]:
        """Run one K-held eval forward using the exact current B3S prefix."""
        import numpy as np

        torch = runtime.modules["torch"]
        pop_robust, arm_common = runtime.modules["pop_robust"], runtime.modules["arm_common"]
        stack_cpu = np.asarray(activity_stack, dtype=np.float32)
        neural_cpu = np.asarray(neural_windows, dtype=np.float32)
        side_cpu = np.asarray(normalized_t4, dtype=np.float32)
        sliced = physically_slice_variable_prefix_held_units(
            neural_cpu, stack_cpu, side_cpu, held_mask,
            expected_prefix_length=expected_prefix_length,
            full_prefix_activity_sha256=expected_prefix_activity_sha256,
        )
        neural = torch.as_tensor(sliced.neural_windows, dtype=torch.float32, device=runtime.device)
        stack = torch.as_tensor(sliced.b3s_activity_stack, dtype=torch.float32, device=runtime.device)
        side = torch.as_tensor(sliced.normalized_t4, dtype=torch.float32, device=runtime.device)
        calibration = stack.unsqueeze(0).expand(neural.shape[0], -1, -1, -1)
        side_batch = side.unsqueeze(0).expand(neural.shape[0], -1, -1)
        _require(calibration.shape[1] == expected_prefix_length,
                 "concrete executor tried a hidden fixed-30 B3S substitution")
        _require(neural.shape[0] >= 1, "concrete executor needs at least one trial endpoint")
        before_model = arm_common.state_sha256(runtime.model)
        rng_before = _source_rng_snapshot(runtime.modules)
        with pop_robust.dynamic_dropout_recorder() as recorder:
            with torch.no_grad():
                first_chunks: list[Any] = []
                for start in range(0, int(neural.shape[0]), 128):
                    stop = min(start + 128, int(neural.shape[0]))
                    first, _ = runtime.model(
                        neural[start:stop], calib_trials=calibration[start:stop], side_features=side_batch[start:stop],
                    )
                    second, _ = runtime.model(
                        neural[start:stop], calib_trials=calibration[start:stop], side_features=side_batch[start:stop],
                    )
                    _require(torch.equal(first, second),
                             "concrete executor repeated held forward drift")
                    first_chunks.append(first)
                first = torch.cat(first_chunks, dim=0)
        rng_after = _source_rng_snapshot(runtime.modules)
        after_model = arm_common.state_sha256(runtime.model)
        _require(bool(torch.isfinite(first).all().item())
                 and before_model == after_model and rng_before == rng_after
                 and recorder["uniform_calls"] == 0 and recorder["dropout_calls"] == [],
                 "concrete executor eval/no-grad/dropout/RNG purity drift")
        velocity = first[:, -1, :].detach().cpu().numpy().astype(np.float64, copy=False)
        _require(velocity.shape == (validity.valid_mask.size, 2),
                 "concrete executor last-bin/validity endpoint shape drift")
        # Restore sealed behavior units before CDM-D displacement integration.
        physical = __import__("src.causal_dual_memory_cell_d_v1.physical", fromlist=["restore_physical_velocity"])
        restored = physical.restore_physical_velocity(
            velocity, behavior_mean=runtime.behavior_normalizer.mean, behavior_std=runtime.behavior_normalizer.std,
        )
        completed = _load_runtime_primitives()[0].CompletedVelocityPrediction(restored, validity)
        evidence = {
            "model_state_before_sha256": before_model,
            "model_state_after_sha256": after_model,
            "rng_before": rng_before, "rng_after": rng_after, "rng_unchanged": True,
            "dynamic_dropout_calls_before": 0, "dynamic_dropout_calls_after": 0,
            "dynamic_dropout_unchanged": True,
            "repeated_outputs_bitwise_equal": True,
            "prefix_length": expected_prefix_length,
            "forward_chunk_count": len(first_chunks),
            "max_endpoints_per_forward_chunk": 128,
            "prefix_activity_sha256": sliced.prefix_activity_sha256,
            "retained_channel_indices": [int(value) for value in sliced.retained_channel_indices],
            "held_channel_indices": [int(value) for value in sliced.held_channel_indices],
        }
        return completed, evidence

    def execute_completed_trial(
        self, *, material: Strict27SessionMaterial, budget: int, trial_id: str,
        support_trial_ids: tuple[str, ...], flags: execute.RuntimeFlags,
    ) -> FinalizedFourGroupPseudo:
        runtime = self._require_runtime()
        core, _adapter, _audit = _load_runtime_primitives()
        import numpy as np

        key = (material.session_id, budget)
        _require(key in runtime.state_by_session_budget, "concrete executor needs fresh session/budget state")
        _require(trial_id in material.trial_views_by_id and trial_id in material.neural_windows_by_id,
                 "concrete executor trial capability/source neural endpoint missing")
        memory = runtime.state_by_session_budget[key]
        if runtime.measurement_started_monotonic <= 0.0:
            import time

            runtime.measurement_started_monotonic = float(time.monotonic())
        prediction_inputs = memory.read_prediction_inputs()
        expected_prefix = budget + int(memory.state.activity.query_count)
        _require(prediction_inputs.activity_trials.shape[0] == expected_prefix
                 and 1 <= expected_prefix <= 30,
                 "concrete executor causal B3S prefix length/state drift")
        expected_prefix_activity_sha256 = _variable_prefix_array_digest(prediction_inputs.activity_trials)
        forward = material.neural_windows_by_id[trial_id]
        _require(isinstance(forward, _ConcreteForwardTrial), "concrete executor neural endpoint type drift")
        normalized = self._normalized_active_t4(runtime, prediction_inputs.active_t4).detach().cpu().numpy()
        views = material.trial_views_by_id[trial_id]
        predictions: list[Any] = []
        group_evidence: list[Mapping[str, object]] = []
        for group in range(execute.GROUP_COUNT):
            completed, evidence = self._torch_variable_prefix_forward(
                runtime, neural_windows=forward.neural_windows,
                activity_stack=prediction_inputs.activity_trials, normalized_t4=normalized,
                held_mask=memory.state.carrier.groups.held_mask(group), validity=views.velocity_validity,
                expected_prefix_length=expected_prefix,
                expected_prefix_activity_sha256=expected_prefix_activity_sha256,
            )
            predictions.append(completed)
            group_evidence.append(evidence)
        flags.model_forward_calls += 2 * sum(int(item["forward_chunk_count"]) for item in group_evidence)
        pending = memory.observe_completed_trial(
            b3s_trial_activity=views.b3s_activity,
            carrier_trial_counts=views.carrier_counts,
            complementary_predictions=tuple(predictions),
        )
        before_activity, before_carrier = memory.state.activity.digest, memory.state.carrier.digest
        if budget == 30:
            # Source-Audit's post-support M30 safety pool is offline only.
            # We may inspect its pseudo directions but deliberately never
            # commit its pending update to deployment carrier/activity state.
            outcome = memory.commit(pending) if False else None
            _require(memory.state.activity.digest == before_activity and memory.state.carrier.digest == before_carrier,
                     "M30 safety-pool physical inference changed deployment state")
            accepted_count = 0
            after_activity, after_carrier = before_activity, before_carrier
        else:
            outcome = memory.commit(pending)
            accepted_count = execute.GROUP_COUNT if outcome.committed else 0
            after_activity, after_carrier = memory.state.activity.digest, memory.state.carrier.digest
        # Each logical chunk is intentionally evaluated twice for the
        # bitwise-repeat proof.  Count both actual model executions in the
        # measured endpoint-throughput numerator; trials remain logical
        # completed-query trials and are counted once below.
        actual_chunks = 2 * sum(int(item["forward_chunk_count"]) for item in group_evidence)
        runtime.forward_calls += actual_chunks
        runtime.endpoint_chunks_completed += actual_chunks
        runtime.completed_trials += 1
        pseudo = tuple(item.theta_index if item.accepted else None for item in pending.pseudo_directions)
        reasons = tuple(None if item.accepted else item.reason for item in pending.pseudo_directions)
        evidence = {
            "all_four_groups_finalized": True, "group_label_broadcast": False,
            "all_group_inputs_physically_sliced": True, "held_group_count": execute.GROUP_COUNT,
            "label_join_before_outcomes": False,
            "deployment_activity_memory_before_sha256": before_activity,
            "deployment_activity_memory_after_sha256": after_activity,
            "carrier_state_before_sha256": before_carrier,
            "carrier_state_after_sha256": after_carrier,
            "accepted_group_update_count": accepted_count,
            "activity_fifo_capacity": int(memory.state.activity.fifo_capacity),
            "activity_query_count_before": expected_prefix - budget,
            "activity_query_count_after": int(memory.state.activity.query_count),
            "m30_audit_enters_deployment_activity_memory": False if budget == 30 else True,
            "variable_prefix_lengths_by_group": [int(row["prefix_length"]) for row in group_evidence],
            "group_forward_evidence": [dict(row) for row in group_evidence],
        }
        # All purity fields must be independently present for every held group;
        # a caller cannot replace an absent proof with a default zero counter.
        for item in group_evidence:
            _require_exact_rng_snapshot(item.get("rng_before"), label="held forward RNG before")
            _require_exact_rng_snapshot(item.get("rng_after"), label="held forward RNG after")
            _require(item.get("rng_unchanged") is True and item.get("dynamic_dropout_unchanged") is True,
                     "held forward RNG/dropout proof missing or non-pure")
        return FinalizedFourGroupPseudo(
            material.session_id, trial_id, pseudo, reasons, prediction_inputs.state_digest, evidence,
        )

    def resources(self) -> Mapping[str, object]:
        runtime = self._require_runtime()
        import os
        import resource
        import time

        torch = runtime.modules["torch"]
        _require(runtime.endpoint_chunks_completed > 0 and runtime.completed_trials > 0
                 and runtime.measurement_started_monotonic > 0.0,
                 "source executor resources require completed measured held-group forwards")
        # Final device readings must follow all queued work.  This is an
        # execution-only call, never reached from the static/no-CUDA route.
        torch.cuda.synchronize(0)
        wall_seconds = float(time.monotonic() - runtime.measurement_started_monotonic)
        _require(wall_seconds > 0.0, "source executor monotonic measured duration must be positive")
        endpoint_chunks_per_s = float(runtime.endpoint_chunks_completed) / wall_seconds
        trials_per_s = float(runtime.completed_trials) / wall_seconds
        _require(endpoint_chunks_per_s > 0.0 and trials_per_s > 0.0,
                 "source executor measured throughput must be positive")
        return {
            "endpoint_chunks_per_s": endpoint_chunks_per_s,
            "trials_per_s": trials_per_s,
            "wall_seconds": wall_seconds,
            "endpoint_chunks_completed": int(runtime.endpoint_chunks_completed),
            "completed_trials": int(runtime.completed_trials),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "current_cuda_allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "current_cuda_reserved_bytes": int(torch.cuda.memory_reserved(0)),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        }

    def close(self) -> None:
        try:
            if self._runtime is not None:
                self._runtime.closed = True
        finally:
            self._restore_route_local_tf32()


def build_reviewed_physical_backend(
    *, root: Path, source_data: execute.StrictSourceDataRootCapability,
    selected_device: Mapping[str, object],
) -> "PhysicalSourceExecutionBackend":
    """Construct the sole route-owned physical provider/executor composition.

    This factory is intentionally side-effect free.  It validates only static
    selected-device/capability identity; source descriptor opens, SWA tensor
    load, CUDA initialization, parser work, and every model forward remain in
    the generic lifecycle after durable attempt publication.
    """
    _validate_pmc_device_profile(selected_device)
    _require(isinstance(source_data, execute.StrictSourceDataRootCapability),
             "reviewed physical factory needs root-bound strict source capability")
    return PhysicalSourceExecutionBackend(
        provider=ConcreteStrict27SessionProvider(root=Path(root), source_data=source_data),
        executor=ConcreteCellDFourGroupExecutor(root=Path(root)),
    )


class PhysicalSourceExecutionBackend:
    """Descriptor-safe composition backend with explicit source/model seams.

    ``build_reviewed_physical_backend`` supplies the sole route-owned concrete
    parser and strict Cell-D executor.  The protocol constructor remains only
    for no-data dependency-injected tests; production cannot inject an
    arbitrary parser/loader through the public CLI or execution API.
    """

    def __init__(self, *, provider: Strict27SessionProvider, executor: CellDFourGroupExecutor) -> None:
        self.provider = provider
        self.executor = executor
        self._preflight_assets: Mapping[str, execute.BoundAsset] | None = None
        self._runtime: _Runtime | None = None
        self._closed = False

    def preflight(self, *, root: Path, identity: execute.SourceExecutionIdentity,
                  flags: execute.RuntimeFlags) -> Mapping[str, object]:
        _require(flags.source_resolved is False and flags.source_opened is False
                 and flags.checkpoint_opened is False and flags.cuda_initialized is False,
                 "physical preflight started after a side effect")
        _validate_pmc_device_profile(identity.selected_device)
        assets = execute.descriptor_read_fixed_assets(root)
        _require(tuple(assets) == tuple(asset.label for asset in execute.FIXED_ASSETS),
                 "physical fixed-asset topology drift")
        _require(execute.parse_strict_train_roster(assets["strict_manifest"].body) == identity.strict_train_roster,
                 "physical strict manifest/identity roster drift")
        self._preflight_assets = assets
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_preflight_v1",
            "fixed_assets": {label: bound.asset.payload() for label, bound in assets.items()},
            "source_resolved_or_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
            "selected_device": dict(identity.selected_device),
            "train_only_provider_not_called": True,
        }

    def prepare(self, *, root: Path, identity: execute.SourceExecutionIdentity,
                flags: execute.RuntimeFlags) -> _Runtime:
        _require(self._preflight_assets is not None, "physical prepare requires fixed-asset preflight")
        _require(flags.source_resolved is False and flags.source_opened is False,
                 "physical source was resolved before durable attempt")
        physical_roster = _physical_roster_for_spec(identity)
        # The provider sees the complete immutable train manifest, but may
        # resolve/open only ``physical_roster``.  In particular the source
        # smoke has no license to construct the other 26 NWB paths merely to
        # make a superficially complete authority table.
        flags.source_resolved = True
        prebind = self.provider.prebind_train_only(
            manifest_roster=identity.strict_train_roster,
            open_roster=physical_roster,
        )
        _require(isinstance(prebind, Mapping)
                 and prebind.get("strict_train_roster") == list(identity.strict_train_roster)
                 and prebind.get("physical_open_roster") == list(physical_roster)
                 and prebind.get("nontrain_path_or_unit_counter_called") is False
                 and prebind.get("val_test_paths_constructed") is False
                 and prebind.get("nonphysical_source_paths_resolved") is False,
                 "strict train-only provider prebind evidence drift")
        sessions = tuple(self.provider.open_train_sessions(roster=physical_roster))
        flags.source_opened = True
        _require(len(sessions) == len(physical_roster)
                 and tuple(item.session_id for item in sessions) == physical_roster,
                 "physical source provider permitted roster/order drift")
        materials = {item.session_id: item for item in sessions}
        _require(tuple(materials) == physical_roster, "physical source material mapping drift")
        # Strict model/SWA loading is delegated to the reviewed executor only
        # after the durable attempt and train-only source bind.  The executor
        # must mark its real checkpoint/CUDA progress in the shared flags.
        load_result = self.executor.load_strict_sealed_swa(
            root=Path(root), swa_bytes=self._preflight_assets["sealed_cell_d_swa"].body,
            selected_device=identity.selected_device, flags=flags,
        )
        _require(isinstance(load_result, Mapping)
                 and set(load_result) == {"runtime_environment", "sealed_swa_load_proof"},
                 "strict sealed Cell-D loader result topology drift")
        runtime_environment = load_result.get("runtime_environment")
        sealed_swa_load_proof = load_result.get("sealed_swa_load_proof")
        execute.validate_runtime_attestation(identity.selected_device, runtime_environment)
        execute.validate_sealed_cell_d_swa_load_proof(sealed_swa_load_proof)
        runtime = _Runtime(
            assets=self._preflight_assets, manifest_roster=identity.strict_train_roster,
            physical_roster=physical_roster, materials=materials,
            provider=self.provider, executor=self.executor, runtime_environment=dict(runtime_environment),
            sealed_swa_load_proof=dict(sealed_swa_load_proof),
            selected_device=dict(identity.selected_device), prebind_evidence=dict(prebind),
        )
        self._runtime = runtime
        return runtime

    def _require_runtime(self, runtime: _Runtime) -> _Runtime:
        _require(runtime is self._runtime and not runtime.closed, "physical source runtime is unavailable")
        return runtime

    @staticmethod
    def _authority_topology(material: Strict27SessionMaterial) -> dict[str, object]:
        core, _source_adapter, _source_audit = _load_runtime_primitives()
        topology = material.source_theta_topology
        _require(isinstance(topology, SourceThetaTopology),
                 "physical source theta topology type drift")
        _require(topology.channel_ids.size == topology.valid_mask.size,
                 "physical source theta topology/channel drift")
        return {
            "session_id": material.session_id,
            "source_descriptor_sha256": material.source_descriptor_sha256,
            "raw_t4_channel_order_sha256": material.raw_t4_channel_order_sha256,
            "theta_valid_mask_sha256": material.theta_valid_mask_sha256,
            "total_unit_count": int(topology.channel_ids.size),
            "valid_unit_count": int(topology.valid_mask.sum()),
            "invalid_unit_count": int((~topology.valid_mask).sum()),
            "valid_mask_sha256": core.array_digest(topology.valid_mask),
            "invalid_assignment_is_minus_one": True,
            "theta_authority_binds_only_validity_not_budget_groups": True,
        }

    def source_authority(self, runtime: _Runtime, *, identity: execute.SourceExecutionIdentity,
                         flags: execute.RuntimeFlags) -> Mapping[str, object]:
        value = self._require_runtime(runtime)
        _require(flags.source_opened and not flags.target_opened and flags.optimizer_steps == 0,
                 "source authority was requested outside source-only state")
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_authority_v1",
            "cell": execute.CELL,
            "identity_sha256": identity.sha256,
            "strict_train_roster": list(value.manifest_roster),
            "strict_train_roster_sha256": execute.roster_sha256(value.manifest_roster),
            "physical_session_count": len(value.physical_roster),
            "strict_manifest_bound_without_nonphysical_resolution": True,
            "normalizers": identity.normalizers.payload(),
            "fixed_assets": {label: bound.asset.payload() for label, bound in value.assets.items()},
            "runtime_environment": dict(value.runtime_environment),
            "sealed_swa_load_proof": dict(value.sealed_swa_load_proof),
            "prebind_evidence": dict(value.prebind_evidence),
            "sessions": [self._authority_topology(value.materials[session]) for session in value.physical_roster],
            "source_only": True,
            "access": {
                "source_opened": True,
                "within_opened": False,
                "external_opened": False,
                "formal_opened": False,
                "target_opened": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "parameter_updates": 0,
                "normalizer_refit": False,
            },
        }

    @staticmethod
    def _audit_authority_for_budget(material: Strict27SessionMaterial, budget: int) -> Any | None:
        _core, source_adapter, _source_audit = _load_runtime_primitives()
        if len(material.ordered_rewarded_trial_ids) < 60:
            return None
        authorities = source_adapter.build_source_audit_authorities(
            material.ordered_rewarded_trial_ids[:60], m4_d_optimal_indices=material.m4_d_optimal_indices,
        )
        return authorities[budget]

    def _finalized_row(
        self,
        *,
        runtime: _Runtime,
        material: Strict27SessionMaterial,
        budget: int,
        trial_id: str,
        support_trial_ids: tuple[str, ...],
        flags: execute.RuntimeFlags,
    ) -> Any:
        core, source_adapter, source_audit = _load_runtime_primitives()
        outcome = runtime.executor.execute_completed_trial(
            material=material, budget=budget, trial_id=trial_id,
            support_trial_ids=support_trial_ids, flags=flags,
        )
        _require(isinstance(outcome, FinalizedFourGroupPseudo)
                 and outcome.session_id == material.session_id and outcome.trial_id == trial_id,
                 "physical Cell-D K=4 outcome identity drift")
        _validate_budget_memory_transition(outcome.physical_evidence, budget=budget)
        # The audit-only label join is purposefully below the finalized K=4
        # outcome check.  The provider method has no path into the executor.
        truth = runtime.provider.join_audit_true_direction(session_id=material.session_id, trial_id=trial_id)
        _require(isinstance(truth, source_adapter.AuditOnlyTrueDirection)
                 and truth.session_id == material.session_id and truth.trial_id == trial_id,
                 "physical audit-only true direction join drift")
        reasons = tuple(outcome.rejection_reasons)
        _require(all(reason is None or isinstance(reason, core.UpdateRejectionReason) for reason in reasons),
                 "physical K=4 pseudo rejection reason type drift")
        return source_audit.GroupedPseudoAuditRow(
            budget=budget, session_id=material.session_id, trial_id=trial_id,
            true_direction=truth, pseudo_direction_indices=outcome.pseudo_direction_indices,
            rejection_reasons=reasons, prefix_digest=outcome.prefix_digest,
        )

    @staticmethod
    def _validate_fresh_budget_state(
        evidence: Mapping[str, object],
        *,
        material: Strict27SessionMaterial,
        budget: int,
        support_trial_ids: tuple[str, ...],
    ) -> None:
        """Require a new dual-memory state for every ``(session, budget)``.

        M30 -> M10 -> M4 is an evaluation order, never a state trajectory;
        neither later budget nor a later source session may inherit a carrier
        or B3S activity stack from the previous cell.
        """
        _require(isinstance(evidence, Mapping), "fresh dual-memory state evidence must be a mapping")
        _require(
            evidence.get("session_id") == material.session_id
            and evidence.get("budget") == budget
            and evidence.get("support_trial_ids") == list(support_trial_ids)
            and evidence.get("fresh_for_session_budget") is True
            and evidence.get("inherits_prior_session_or_budget_state") is False,
            "dual-memory session/budget reset evidence drift",
        )
        _require_sha(evidence.get("initial_activity_memory_sha256"), "initial activity-memory SHA")
        _require_sha(evidence.get("initial_carrier_state_sha256"), "initial carrier-state SHA")
        _require(
            evidence.get("initial_carrier_recipe") == "fixed_ridge_by_trial"
            and evidence.get("initial_carrier_support_rows") == budget
            and evidence.get("raw_m30_t4_used_as_initializer") is False
            and evidence.get("initial_support_rate_domain") == execute.INITIAL_SUPPORT_RATE_DOMAIN
            and evidence.get("online_update_rate_domain") == execute.ONLINE_UPDATE_RATE_DOMAIN
            and isinstance(evidence.get("initial_carrier_parity"), Mapping)
            and evidence["initial_carrier_parity"].get("mode") == "fixed_ridge_by_trial",
            "budget support-only fixed-ridge initializer evidence drift",
        )
        for key in (
            "initial_support_rates_sha256",
            "initial_support_exposure_seconds_sha256",
            "initial_carrier_sha256",
            "budget_groups_sha256",
            "budget_group_assignment_sha256",
            "budget_group_valid_mask_sha256",
        ):
            _require_sha(evidence.get(key), f"budget initial carrier {key}")
        if budget == 30:
            _require(
                evidence.get("activity_fifo_capacity") == 0
                and evidence.get("initial_activity_query_count") == 0,
                "fresh M30 state did not begin with zero-capacity activity FIFO",
            )

    def _session_b8(self, runtime: _Runtime, *, budget: int, session: str,
                    flags: execute.RuntimeFlags) -> dict[str, object]:
        _core, _source_adapter, source_audit = _load_runtime_primitives()
        material = runtime.materials[session]
        authority = self._audit_authority_for_budget(material, budget)
        topology = self._authority_topology(material)
        if authority is None:
            return {
                "schema": "causal_dual_memory_cell_d_source_execution_session_b8_v1",
                "budget": budget, "session": session, "status": "STOP_MISSING_REQUIRED_CHRONOLOGY",
                "pass": False, "missing_required_chronology_count": 60 - len(material.ordered_rewarded_trial_ids),
                "unit_topology": topology, "source_authority_unit_topology": topology,
                "source_only": True, "target_optimizer_backward_update": 0,
            }
        support = authority.deployment.support_trial_ids
        reset = runtime.executor.begin_session_budget(
            material=material, budget=budget, support_trial_ids=support, flags=flags,
        )
        self._validate_fresh_budget_state(
            reset, material=material, budget=budget, support_trial_ids=support,
        )
        groups = runtime.executor.groups_for_session_budget(material=material, budget=budget)
        core, _source_adapter, _source_audit = _load_runtime_primitives()
        _require(isinstance(groups, core.ComplementaryGroups)
                 and groups.channel_ids.shape == material.source_theta_topology.channel_ids.shape
                 and (groups.channel_ids == material.source_theta_topology.channel_ids).all()
                 and (groups.valid_mask == material.source_theta_topology.valid_mask).all(),
                 "budget-specific live group map/source theta topology drift")
        rows = tuple(self._finalized_row(
            runtime=runtime, material=material, budget=budget, trial_id=trial_id,
            support_trial_ids=support, flags=flags,
        ) for trial_id in authority.audit_trial_ids)
        views = tuple(material.trial_views_by_id[trial_id] for trial_id in authority.audit_trial_ids)
        inputs = source_audit.carrier_b8_inputs_from_finalized_rows(
            rows, views, audit_authority=authority, groups=groups,
        )
        evidence = dict(source_audit.carrier_b8_aggregate(inputs))
        if evidence["status"] == "COMPLETE_FIXED_POOL" and evidence["defined_units_correct"] == 0:
            evidence["status"] = "STOP_NO_DEFINED_VALID_COSINES"
            evidence["pass"] = False
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_session_b8_v1",
            "budget": budget,
            "session": session,
            "status": evidence["status"],
            "pass": bool(evidence["pass"]),
            "unit_topology": dict(evidence["unit_topology"]),
            "source_authority_unit_topology": topology,
            "grouped_b8": evidence,
            "fixed_pool_trial_ids": list(authority.audit_trial_ids),
            "support_trial_ids": list(support),
            "audit_enters_deployment_memory": authority.audit_enters_deployment_memory,
            "budget_groups_sha256": groups.digest,
            "budget_group_assignment_sha256": core.array_digest(groups.assignment),
            "budget_group_valid_mask_sha256": core.array_digest(groups.valid_mask),
            "budget_initial_carrier_recipe": reset.get("initial_carrier_recipe"),
            "budget_initial_carrier_sha256": reset.get("initial_carrier_sha256"),
            "budget_initial_carrier_support_rows": reset.get("initial_carrier_support_rows"),
            "budget_initial_carrier_parity": dict(reset["initial_carrier_parity"]),
            "initial_support_rate_domain": reset.get("initial_support_rate_domain"),
            "online_update_rate_domain": reset.get("online_update_rate_domain"),
            "initial_support_rates_sha256": reset.get("initial_support_rates_sha256"),
            "initial_support_exposure_seconds_sha256": reset.get("initial_support_exposure_seconds_sha256"),
            "raw_m30_t4_used_as_initializer": reset.get("raw_m30_t4_used_as_initializer"),
            "source_only": True,
            "target_optimizer_backward_update": 0,
        }

    def run_smoke(self, runtime: _Runtime, *, identity: execute.SourceExecutionIdentity,
                  flags: execute.RuntimeFlags) -> Mapping[str, object]:
        value = self._require_runtime(runtime)
        material = value.materials["sub-C_ses-CO-20131003"]
        authority = self._audit_authority_for_budget(material, 30)
        _require(authority is not None, "fixed smoke requires first 60 rewarded source trials")
        expected_ids = tuple(material.ordered_rewarded_trial_ids[position] for position in (30, 31))
        _require(expected_ids == authority.audit_trial_ids[:2], "fixed smoke M30 positions 30--31 drift")
        reset = value.executor.begin_session_budget(
            material=material, budget=30, support_trial_ids=authority.deployment.support_trial_ids, flags=flags,
        )
        self._validate_fresh_budget_state(
            reset, material=material, budget=30,
            support_trial_ids=authority.deployment.support_trial_ids,
        )
        rows = tuple(self._finalized_row(
            runtime=value, material=material, budget=30, trial_id=trial_id,
            support_trial_ids=authority.deployment.support_trial_ids, flags=flags,
        ) for trial_id in expected_ids)
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_smoke_v1",
            "session": material.session_id,
            "budget": 30,
            "support_positions": list(range(30)),
            "audit_positions": [30, 31],
            "audit_trial_ids": list(expected_ids),
            "group_count": execute.GROUP_COUNT,
            "all_four_groups_finalized": all(
                len(row.pseudo_direction_indices) == execute.GROUP_COUNT for row in rows
            ),
            "b8_threshold_applied": False,
            "budget_initial_carrier_recipe": reset.get("initial_carrier_recipe"),
            "budget_initial_carrier_support_rows": reset.get("initial_carrier_support_rows"),
            "raw_m30_t4_used_as_initializer": reset.get("raw_m30_t4_used_as_initializer"),
            "initial_support_rate_domain": reset.get("initial_support_rate_domain"),
            "online_update_rate_domain": reset.get("online_update_rate_domain"),
            "initial_support_rates_sha256": reset.get("initial_support_rates_sha256"),
            "initial_support_exposure_seconds_sha256": reset.get("initial_support_exposure_seconds_sha256"),
            "budget_initial_carrier_parity": dict(reset["initial_carrier_parity"]),
            "budget_initial_carrier_sha256": reset.get("initial_carrier_sha256"),
            "budget_groups_sha256": reset.get("budget_groups_sha256"),
            "budget_group_assignment_sha256": reset.get("budget_group_assignment_sha256"),
            "budget_group_valid_mask_sha256": reset.get("budget_group_valid_mask_sha256"),
            "unit_topology": self._authority_topology(material),
            "source_only": True,
            "target_optimizer_backward_update": 0,
        }

    def run_budget(self, runtime: _Runtime, *, budget: int, identity: execute.SourceExecutionIdentity,
                   flags: execute.RuntimeFlags) -> Sequence[Mapping[str, object]]:
        value = self._require_runtime(runtime)
        _require(budget in execute.FAIL_FAST_BUDGET_ORDER, "physical source gate budget drift")
        _require(value.physical_roster == value.manifest_roster,
                 "full source gate may not run a partial physical source roster")
        return tuple(self._session_b8(value, budget=budget, session=session, flags=flags)
                     for session in value.manifest_roster)

    def resources(self, runtime: _Runtime, *, flags: execute.RuntimeFlags) -> Mapping[str, object]:
        value = self._require_runtime(runtime)
        resource = dict(value.executor.resources())
        resource["selected_device"] = dict(value.selected_device)
        # ``resources`` must report the frozen profile, while the executor may
        # return extra timing counters.  The core validator rejects missing or
        # incoherent peak/current values.
        return resource

    def revalidate(self, *, root: Path, identity: execute.SourceExecutionIdentity,
                   flags: execute.RuntimeFlags) -> None:
        value = self._require_runtime(self._runtime)  # type: ignore[arg-type]
        reloaded = execute.descriptor_read_fixed_assets(root)
        _require({label: bound.asset.payload() for label, bound in reloaded.items()}
                 == {label: bound.asset.payload() for label, bound in value.assets.items()},
                 "physical fixed assets drifted during source execution")
        _require(execute.parse_strict_train_roster(reloaded["strict_manifest"].body) == value.manifest_roster,
                 "physical strict train roster drifted during source execution")

    def close(self, runtime: _Runtime | None) -> None:
        value = runtime if isinstance(runtime, _Runtime) else self._runtime
        if self._closed:
            return
        self._closed = True
        if value is not None:
            value.closed = True
        try:
            self.executor.close()
        finally:
            self.provider.close()
