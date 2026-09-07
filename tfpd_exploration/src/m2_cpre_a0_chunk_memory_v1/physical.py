"""Admission, static contracts, and pure receipt assembly for M2 A0.

No dataset/checkpoint/Torch import occurs at module import.  The only public
runner path is dry/inert; a future root review must retain an in-process
capability object before metadata or model runtime factories can be invoked.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import chunk_memory, inventory, plan, receipts


class PhysicalContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalContractError(message)


def _canonical_json_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256_literal(value: str, *, field: str) -> None:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"{field} must be an exact lowercase SHA-256 literal")


def source_closure(repo_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in plan.BOUND_PATTERNS:
        path = repo_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure path missing/nonregular: {relative}")
        body = path.read_bytes()
        files[relative] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    _require(set(files) == set(plan.BOUND_PATTERNS), "closure key drift")
    return {
        "files": files,
        "closure_sha256": _canonical_json_sha(files),
        "workorder_sha256": files[plan.WORKORDER_RELATIVE]["sha256"],
        "cpre_v2_repair_sha256": files[plan.CPRE_V2_REPAIR_RELATIVE]["sha256"],
        "historical_cpre_binding_sha256": files[plan.HISTORICAL_CPRE_BINDING_RELATIVE]["sha256"],
        "design_review_sha256": files[plan.DESIGN_REVIEW_RELATIVE]["sha256"],
    }


def assert_bound_review(closure: Mapping[str, Any]) -> None:
    _require(closure.get("workorder_sha256") == plan.WORKORDER_SHA256,
             "bound workorder bytes drift")
    _require(closure.get("design_review_sha256") == plan.DESIGN_REVIEW_SHA256,
             "bound design-review bytes drift")
    _require(closure.get("historical_cpre_binding_sha256") == plan.HISTORICAL_CPRE_BINDING_SHA256,
             "bound historical C-Pre binding workorder bytes drift")


def assert_bound_cpre_v2_repair(closure: Mapping[str, Any]) -> None:
    """Require both the frozen parent WO and the narrow V2 repair addendum."""
    assert_bound_review(closure)
    _require(closure.get("cpre_v2_repair_sha256") == plan.CPRE_V2_REPAIR_SHA256,
             "bound C-Pre V2 repair workorder bytes drift")


@dataclass(frozen=True)
class A0ExecutionProfile:
    """Immutable route identity for the one reviewed A0 replay loop.

    A successor may change admission/root/receipt identity without patching
    ``plan`` or duplicating numerical replay.  Science helpers remain shared.
    """
    route: str
    roots: Mapping[str, str]
    attempt_schema: str
    launch_schema: str
    terminal_schema: str
    failure_schema: str
    closure_builder: Callable[[Path], Mapping[str, Any]]
    closure_validator: Callable[[Mapping[str, Any]], None]
    cpre_uses_current_closure: bool = True
    capability_consumer: Callable[..., Mapping[str, Any]] | None = None
    final_validator: Callable[..., None] | None = None

    def root_for(self, surface: str) -> str:
        _require(surface in ("external_post30_local", "within_post30"), "unknown A0 profile surface")
        value = self.roots.get(surface)
        _require(isinstance(value, str), "A0 profile missing shard root")
        return value


V1_EXECUTION_PROFILE = A0ExecutionProfile(
    route=plan.ROUTE_SCHEMA, roots=plan.A0_ROOTS,
    attempt_schema="m2_a0_attempt_v1", launch_schema="m2_a0_launch_v1",
    terminal_schema="m2_a0_terminal_v1", failure_schema="m2_a0_failure_v1",
    closure_builder=source_closure, closure_validator=assert_bound_review,
)


@dataclass(frozen=True)
class DeviceAttestation:
    physical_index: int
    cuda_visible_devices: str
    uuid: str
    logical_device: str
    cuda_initialized: bool

    def validate(self) -> None:
        _require(self.physical_index == 0 and self.cuda_visible_devices == "0"
                 and self.uuid == plan.GPU0_UUID and self.logical_device == "cuda:0",
                 "M2 A0 is physically bound to GPU0 only")
        _require(not self.cuda_initialized, "pre-runtime A0 attestation must not initialize CUDA")


def static_device_profile() -> dict[str, Any]:
    profile = plan.StaticDeviceProfile()
    return {
        "physical_index": profile.physical_index,
        "cuda_visible_devices": profile.cuda_visible_devices,
        "uuid": profile.uuid,
        "logical_device": profile.logical_device,
        "gpu1_forbidden": profile.gpu1_forbidden,
    }


def normalize_gpu0_uuid(observed: object) -> dict[str, str]:
    """Canonicalize only PyTorch's optional transport-prefix omission.

    CUDA may expose the physical UUID as either ``GPU-<uuid>`` or ``<uuid>``.
    These are the sole two accepted spellings; this is not a fuzzy UUID match.
    """
    raw = str(observed)
    without_prefix = plan.GPU0_UUID.removeprefix("GPU-")
    _require(raw in (plan.GPU0_UUID, without_prefix),
             "visible A0 CUDA device UUID is not frozen physical GPU0")
    return {"raw_uuid": raw, "canonical_uuid": plan.GPU0_UUID}


def scheduler_profile(surface: str) -> dict[str, Any]:
    _require(surface in plan.SCHEDULER_PROFILES, "unknown scheduler surface")
    profile = plan.SCHEDULER_PROFILES[surface]
    return {
        "surface": profile.surface,
        "cpu_affinity": list(profile.cpu_affinity),
        "num_workers": profile.workers,
        "OMP_NUM_THREADS": profile.omp_threads,
        "MKL_NUM_THREADS": profile.mkl_threads,
        "OPENBLAS_NUM_THREADS": profile.openblas_threads,
        "NUMEXPR_NUM_THREADS": profile.numexpr_threads,
        "gpu1_query_or_fallback_forbidden": True,
    }


def attest_a0_runtime_scheduler(*, surface: str, environ: Mapping[str, str] | None = None,
                                 affinity: Iterable[int] | None = None, pid: int | None = None) -> dict[str, Any]:
    """Fail closed unless the current process is in its frozen M2 envelope.

    This deliberately inspects only this process, its declared environment and
    its CPU mask.  It neither enumerates nor touches physical GPU1.
    """
    expected = scheduler_profile(surface)
    environment = os.environ if environ is None else environ
    observed_affinity = set(os.sched_getaffinity(0) if affinity is None else affinity)
    _require(observed_affinity == set(expected["cpu_affinity"]), "A0 process CPU-affinity drift")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _require(environment.get(name) == expected[name], f"A0 process {name} drift")
    _require(environment.get("CUDA_VISIBLE_DEVICES") == "0"
             and environment.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
             "A0 process CUDA visibility/order drift")
    return {
        "surface": surface, "pid": int(os.getpid() if pid is None else pid),
        "cpu_affinity": sorted(observed_affinity),
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "num_workers": 0,
    }


class _Capability:
    __slots__ = ("_token", "_binding", "_consumed")

    def __init__(self, token: object, binding: Mapping[str, Any]) -> None:
        self._token = token
        self._binding = dict(binding)
        self._consumed = False


_ISSUER_TOKEN = object()


def _canonical_result_path(repo_root: Path, relative: str) -> Path:
    _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
             "result root relative path must stay within the repository")
    return repo_root / relative


def _prospective_root_witness(*, repo_root: Path, relative: str) -> dict[str, Any]:
    """Bind a future live route to one absent named canonical result root."""
    root = _canonical_result_path(repo_root, relative)
    parent = root.parent
    _require(parent.is_dir() and not parent.is_symlink() and not root.exists(),
             "live capability requires an absent canonical root under a regular parent")
    info = os.stat(parent, follow_symlinks=False)
    return {"root_relative": relative, "root_name": root.name,
            "parent_device": int(info.st_dev), "parent_inode": int(info.st_ino)}


def _verify_prospective_root_witness(*, witness: Mapping[str, Any], repo_root: Path,
                                     expected_relative: str, supplied_root: Path) -> None:
    _require(set(witness) == {"root_relative", "root_name", "parent_device", "parent_inode"}
             and witness.get("root_relative") == expected_relative,
             "live capability root-witness schema/relative drift")
    expected = _canonical_result_path(repo_root, expected_relative)
    _require(supplied_root.absolute() == expected.absolute() and expected.name == witness.get("root_name"),
             "live executor received a noncanonical result root")
    parent = expected.parent
    info = os.stat(parent, follow_symlinks=False)
    _require(parent.is_dir() and not parent.is_symlink()
             and int(info.st_dev) == int(witness.get("parent_device", -1))
             and int(info.st_ino) == int(witness.get("parent_inode", -1))
             and not expected.exists(),
             "live canonical result parent/root freshness drift")


def _capture_reserved_root(root: Path) -> dict[str, int]:
    info = os.stat(root, follow_symlinks=False)
    _require(stat.S_ISDIR(info.st_mode) and not root.is_symlink(),
             "reserved result root is not a regular directory")
    return {"device": int(info.st_dev), "inode": int(info.st_ino)}


def _verify_reserved_root(root: Path, reservation: Mapping[str, int]) -> None:
    info = os.stat(root, follow_symlinks=False)
    _require(stat.S_ISDIR(info.st_mode) and not root.is_symlink()
             and int(info.st_dev) == int(reservation.get("device", -1))
             and int(info.st_ino) == int(reservation.get("inode", -1)),
             "reserved canonical result root was replaced")


def _mint_test_capability(binding: Mapping[str, Any]) -> _Capability:
    """Test-only opaque mint.  Public CLI never receives this symbol."""
    return _Capability(_ISSUER_TOKEN, binding)


def bridge_admitted_profile_capability(binding: Mapping[str, Any]) -> object:
    """Create a V1-loop capability only after a route-specific profile admitted it.

    This is not a public CLI mint: the caller must be an in-process execution
    profile that has already consumed and revalidated its own opaque admission
    token.  Keeping the bridge here preserves the single replay loop's token
    type without allowing a successor to use the test mint.
    """
    _require(binding.get("mode") == "profile_admitted", "profile bridge requires admitted binding")
    return _Capability(_ISSUER_TOKEN, binding)


def _consume(capability: object, *, expected_root: str, closure: Mapping[str, Any],
             repo_root: Path | None = None, supplied_root: Path | None = None) -> Mapping[str, Any]:
    _require(isinstance(capability, _Capability) and capability._token is _ISSUER_TOKEN,
             "opaque in-process capability required")
    _require(not capability._consumed, "capability is one-shot and was already consumed")
    binding = capability._binding
    _require(binding.get("root_relative") == expected_root, "capability root identity drift")
    _require(binding.get("closure_sha256") == closure.get("closure_sha256"), "capability closure drift")
    witness = binding.get("root_witness")
    if witness is not None:
        _require(repo_root is not None and supplied_root is not None and isinstance(witness, Mapping),
                 "live root-bound capability requires canonical executor root evidence")
        _verify_prospective_root_witness(witness=witness, repo_root=repo_root,
                                         expected_relative=expected_root, supplied_root=supplied_root)
    capability._consumed = True
    return binding


def _consume_a0(capability: object, *, surface: str, closure: Mapping[str, Any],
                cpre_terminal_sha256: str, roster: Sequence[str], repo_root: Path | None = None,
                supplied_root: Path | None = None, root_relative: str | None = None) -> Mapping[str, Any]:
    expected_root = plan.A0_ROOTS[surface] if root_relative is None else root_relative
    binding = _consume(capability, expected_root=expected_root, closure=closure,
                       repo_root=repo_root, supplied_root=supplied_root)
    _require(binding.get("surface") == surface and tuple(binding.get("roster", ())) == tuple(roster),
             "A0 capability surface/roster drift")
    _require(binding.get("cpre_terminal_sha256") == cpre_terminal_sha256,
             "A0 capability C-Pre terminal drift")
    _require(binding.get("device_profile") == static_device_profile(),
             "A0 capability device profile drift")
    _require(binding.get("scheduler_profile") == scheduler_profile(surface),
             "A0 capability scheduler profile drift")
    _require(binding.get("frozen_decode_batch_size") == plan.A0_FROZEN_DECODE_BATCH_SIZE,
             "A0 capability frozen batch law drift")
    if binding.get("mode") == "live":
        _require(binding.get("resolved_window_size") == plan.WINDOW_BINS
                 and isinstance(binding.get("static_anchor_payload"), Mapping)
                 and binding.get("static_anchor_payload_sha256")
                 == _canonical_json_sha(binding["static_anchor_payload"])
                 and isinstance(binding.get("query_window_authority"), Mapping)
                 and binding.get("query_window_authority_sha256")
                 == _canonical_json_sha(binding["query_window_authority"]),
                 "A0 live capability held input authority drift")
    return binding


def _future_predecessors(repo_root: Path) -> dict[str, Any]:
    """Deferred descriptor validation; never invoked by dry/no-data tests."""
    payloads: dict[str, Mapping[str, Any]] = {}
    witnesses: dict[str, Any] = {}
    for name, (relative, expected_sha) in plan.PREDECESSOR_TERMINALS.items():
        path = repo_root / relative
        descriptor = receipts.descriptor_read(path.parent, path.name, expected_sha256=expected_sha)
        payloads[name] = descriptor.payload
        witnesses[name] = {"relative": relative, "sha256": descriptor.sha256}
    receipts.semantic_predecessor_validator(payloads)
    return witnesses


def _validate_v1_cpre_failure_payloads(*, attempt: Mapping[str, Any], failure: Mapping[str, Any]) -> None:
    """Semantic half of the immutable V1 C-Pre failure predecessor codec."""
    _require(attempt.get("schema") == "m2_cpre_attempt_v1"
             and attempt.get("status") == "ATTEMPT_RESERVED_BEFORE_METADATA"
             and attempt.get("route") == plan.ROUTE_SCHEMA
             and attempt.get("root_relative") == plan.CPRE_ROOT_RELATIVE
             and attempt.get("target_access") is False
             and attempt.get("no_model_checkpoint_cuda_r2_or_target_read_before_inventory") is True,
             "V1 C-Pre attempt semantic drift")
    _require(failure.get("schema") == "m2_cpre_failure_v1" and failure.get("status") == "FAIL_CLOSED"
             and failure.get("attempt_sha256") == plan.CPRE_V1_FAILURE_ATTEMPT_SHA256
             and failure.get("stage") == "metadata_provider"
             and failure.get("error") == "InventoryError: within post30 anchor window count/digest drift",
             "V1 C-Pre failure semantic/stage drift")
    progress = failure.get("progress")
    _require(isinstance(progress, Mapping)
             and progress.get("target_access") is False and progress.get("target_values_read") is False
             and progress.get("model_or_checkpoint_opened") is False and progress.get("cuda_initialized") is False
             and progress.get("metadata_inventory_published") is False,
             "V1 C-Pre failure no-access/progress drift")


def _future_v1_cpre_failure_predecessor(repo_root: Path) -> dict[str, Any]:
    """Held-FD/O_NOFOLLOW admission proof for V2's one immutable V1 failure."""
    root = _canonical_result_path(repo_root, plan.CPRE_ROOT_RELATIVE)
    descriptors = receipts.verify_topology(root, bodies=("attempt.json", "failure.json"))
    attempt = descriptors["attempt.json"]
    failure = descriptors["failure.json"]
    _require(attempt.sha256 == plan.CPRE_V1_FAILURE_ATTEMPT_SHA256
             and failure.sha256 == plan.CPRE_V1_FAILURE_SHA256,
             "V1 C-Pre predecessor exact body SHA drift")
    _validate_v1_cpre_failure_payloads(attempt=attempt.payload, failure=failure.payload)
    return {"root_relative": plan.CPRE_ROOT_RELATIVE,
            "attempt_sha256": attempt.sha256, "failure_sha256": failure.sha256}


def _future_cpre_completion(*, repo_root: Path, expected_terminal_sha256: str | None = None,
                             closure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Descriptor-only C-Pre witness used by the root-only A0 issuer."""
    root = _canonical_result_path(repo_root, plan.CPRE_V2_ROOT_RELATIVE)
    descriptors = receipts.verify_topology(root, bodies=(
        "attempt.json", "metadata_inventory.json", "terminal.json"))
    terminal = descriptors["terminal.json"].payload
    attempt = descriptors["attempt.json"].payload
    _require(descriptors["attempt.json"].sha256 == plan.CPRE_V2_HISTORICAL_ATTEMPT_SHA256
             and descriptors["metadata_inventory.json"].sha256 == plan.CPRE_V2_HISTORICAL_INVENTORY_SHA256
             and descriptors["terminal.json"].sha256 == plan.CPRE_V2_HISTORICAL_TERMINAL_SHA256,
             "accepted historical C-Pre V2 body SHA drift")
    _require(terminal.get("schema") == "m2_cpre_terminal_v2" and terminal.get("status") == "TERMINAL"
             and terminal.get("attempt_sha256") == descriptors["attempt.json"].sha256
             and terminal.get("metadata_inventory_sha256") == descriptors["metadata_inventory.json"].sha256
             and terminal.get("target_access") is False and terminal.get("target_values_read") is False
             and terminal.get("model_or_checkpoint_opened") is False and terminal.get("cuda_initialized") is False
             and terminal.get("v1_failure_predecessor") == {
                 "root_relative": plan.CPRE_ROOT_RELATIVE,
                 "attempt_sha256": plan.CPRE_V1_FAILURE_ATTEMPT_SHA256,
                 "failure_sha256": plan.CPRE_V1_FAILURE_SHA256,
             },
             "C-Pre terminal semantic/no-runtime contract drift")
    _require(attempt.get("schema") == "m2_cpre_attempt_v2"
             and attempt.get("status") == "ATTEMPT_RESERVED_BEFORE_METADATA"
             and attempt.get("root_relative") == plan.CPRE_V2_ROOT_RELATIVE
             and attempt.get("v1_failure_predecessor") == terminal.get("v1_failure_predecessor"),
             "historical C-Pre attempt semantic/predecessor drift")
    _require(_future_v1_cpre_failure_predecessor(repo_root) == terminal.get("v1_failure_predecessor"),
             "historical C-Pre V1 failed predecessor revalidation drift")
    if expected_terminal_sha256 is not None:
        _require(descriptors["terminal.json"].sha256 == expected_terminal_sha256,
                 "A0 requested C-Pre terminal SHA drift")
    historical_closure = terminal.get("source_closure")
    _require(isinstance(historical_closure, Mapping)
             and historical_closure.get("closure_sha256") == plan.CPRE_V2_HISTORICAL_CLOSURE_SHA256,
             "historical C-Pre execution closure drift")
    # ``closure`` deliberately belongs to the *current A0* executable.  It is
    # never compared to the immutable CPU-only C-Pre closure: a source-only
    # transport repair must not fabricate a need to rerun C-Pre.
    if closure is not None:
        assert_bound_review(closure)
    metadata = descriptors["metadata_inventory.json"].payload
    _require(metadata.get("schema") == "m2_cpre_metadata_inventory_v2"
             and metadata.get("target_values_read") is False
             and metadata.get("window_coordinate_law") == "endpoint_s_ge_raw_trial_start_n_plus_49",
             "C-Pre metadata inventory semantic drift")
    return {"terminal_sha256": descriptors["terminal.json"].sha256,
            "metadata_inventory_sha256": descriptors["metadata_inventory.json"].sha256,
            "historical_cpre_closure_sha256": plan.CPRE_V2_HISTORICAL_CLOSURE_SHA256,
            "metadata": metadata}


def _future_static_anchor(repo_root: Path) -> tuple[dict[str, Any], str]:
    path = _canonical_result_path(repo_root, plan.ANCHOR_SCORE_RELATIVE)
    descriptor = receipts.descriptor_read(path.parent, path.name, expected_sha256=plan.ANCHOR_SCORE_SHA256)
    # Validate the exact historical body, but retain that *full* descriptor
    # payload for the later live A0 input-authority codec.  Returning the
    # reduced summary here would make its second schema validation fail after
    # a costly replay, because ``build_a0_input_authority`` correctly expects
    # the score receipt's schema/status/rows fields.
    validate_static_anchor(descriptor.payload)
    return dict(descriptor.payload), descriptor.sha256


def _future_anchor_window_authority(*, anchor_payload: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract only pre-attempt facts that actually exist in the anchor score.

    The historical score has count plus ``core.array_sha256`` for each ordered
    window sequence.  It deliberately does *not* contain C-Pre's int64 digest,
    which must only be computed after C-Pre has published its attempt and read
    metadata.  Keeping those two encodings separate preserves attempt-first.
    """
    rows = anchor_payload.get("rows")
    _require(isinstance(rows, list), "anchor score rows absent for C-Pre authority")
    authority: dict[str, dict[str, dict[str, Any]]] = {surface: {} for surface in plan.SURFACE_ORDER}
    for surface, anchor_surface in (("external_post30_local", "external_official_query"),
                                    ("within_post30", "within_post30")):
        matching = [row for row in rows if isinstance(row, Mapping)
                    and row.get("cell") == plan.ANCHOR_CELL and row.get("surface") == anchor_surface]
        _require(bool(matching), "anchor score has no activity30/M10 surface rows")
        for row in matching:
            session = row.get("session")
            count = int(row.get("window_count", -1))
            digest = str(row.get("ordered_window_starts_sha256", ""))
            _require(isinstance(session, str) and bool(session) and count > 0,
                     "anchor score session/count schema drift")
            _sha256_literal(digest, field="anchor_score_ordered_window_starts_sha256")
            _require(session not in authority[surface], "anchor score has duplicate session window rows")
            if surface == "external_post30_local":
                authority[surface][session] = {
                    "anchor_full_window_count": count,
                    "anchor_full_window_anchor_core_sha256": digest,
                }
            else:
                authority[surface][session] = {
                    "anchor_post30_window_count": count,
                    "anchor_post30_window_anchor_core_sha256": digest,
                }
    return authority


def _derive_a0_query_window_authority(*, metadata: Mapping[str, Any], anchor_payload: Mapping[str, Any],
                                      surface: str, roster: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Derive A0's immutable local post30 surface from the completed C-Pre body."""
    crosschecks = metadata.get("anchor_window_crosschecks", {})
    _require(isinstance(crosschecks, Mapping) and isinstance(crosschecks.get(surface), Mapping),
             "C-Pre lacks the bound A0 surface window crosscheck")
    output: dict[str, dict[str, Any]] = {}
    anchor_surface = "external_official_query" if surface == "external_post30_local" else "within_post30"
    anchor_rows = anchor_payload.get("rows")
    _require(isinstance(anchor_rows, list), "static anchor score rows absent for A0 window authority")
    for session in roster:
        current = crosschecks[surface].get(session)
        _require(isinstance(current, Mapping), "C-Pre A0 roster/window authority drift")
        required = ("post30_window_count", "post30_window_starts_sha256", "post30_window_anchor_core_sha256",
                    "full_window_count", "full_window_starts_sha256", "full_window_anchor_core_sha256",
                    "post30_is_deterministic_subset_of_full")
        _require(all(field in current for field in required)
                 and current.get("post30_is_deterministic_subset_of_full") is True,
                 "C-Pre A0 window crosscheck schema drift")
        matching = [row for row in anchor_rows if isinstance(row, Mapping)
                    and row.get("cell") == plan.ANCHOR_CELL
                    and row.get("surface") == anchor_surface and row.get("session") == session]
        _require(len(matching) == 1, "static anchor lacks unique activity30/M10 session window authority")
        anchor_row = matching[0]
        _sha256_literal(str(anchor_row.get("ordered_window_starts_sha256", "")),
                        field="anchor_score_ordered_window_starts_sha256")
        anchor_count = int(anchor_row.get("window_count", -1))
        _require(anchor_count > 0, "static anchor score window count drift")
        if surface == "external_post30_local":
            _require(anchor_count == int(current["full_window_count"])
                     and anchor_row["ordered_window_starts_sha256"] == current["full_window_anchor_core_sha256"],
                     "external C-Pre full window authority does not match exact anchor score row")
            output[session] = {
                "cpre_anchor_reference_int64_sha256": str(current["full_window_starts_sha256"]),
                "anchor_score_ordered_window_starts_sha256": str(current["full_window_anchor_core_sha256"]),
                "anchor_reference_window_count": anchor_count,
                "within_post30_exact_anchor_parity": None,
            }
        else:
            _require(anchor_count == int(current["post30_window_count"])
                     and anchor_row["ordered_window_starts_sha256"] == current["post30_window_anchor_core_sha256"],
                     "within C-Pre post30 window authority does not match exact anchor score row")
            output[session] = {
                "cpre_anchor_reference_int64_sha256": str(current["post30_window_starts_sha256"]),
                "anchor_score_ordered_window_starts_sha256": str(current["post30_window_anchor_core_sha256"]),
                "anchor_reference_window_count": anchor_count,
                "within_post30_exact_anchor_parity": True,
            }
    return output


def _path_map_sha256(paths_by_surface: Mapping[str, Sequence[Path]]) -> str:
    _require(set(paths_by_surface) == set(plan.SURFACE_ORDER), "C-Pre path-map surface topology drift")
    canonical = {surface: [str(Path(path)) for path in paths_by_surface[surface]]
                 for surface in plan.SURFACE_ORDER}
    _require(all(canonical[surface] for surface in plan.SURFACE_ORDER), "C-Pre path-map has an empty surface")
    return _canonical_json_sha(canonical)


def _issue_live_cpre_capability(*, repo_root: Path, root: Path,
                                paths_by_surface: Mapping[str, Sequence[Path]],
                                resolved_metadata_facts: Mapping[str, object]) -> _Capability:
    """Root-only, descriptor-first live issuer.  Public CLI cannot call it."""
    closure = source_closure(repo_root)
    assert_bound_review(closure)
    predecessors = _future_predecessors(repo_root)
    facts = inventory.validate_resolved_metadata_facts(resolved_metadata_facts)
    anchor_payload, anchor_descriptor_sha = _future_static_anchor(repo_root)
    anchor_window_authority = _future_anchor_window_authority(anchor_payload=anchor_payload)
    witness = _prospective_root_witness(repo_root=repo_root, relative=plan.CPRE_ROOT_RELATIVE)
    _verify_prospective_root_witness(witness=witness, repo_root=repo_root,
                                     expected_relative=plan.CPRE_ROOT_RELATIVE, supplied_root=root)
    return _Capability(_ISSUER_TOKEN, {
        "mode": "live", "root_relative": plan.CPRE_ROOT_RELATIVE, "root_witness": witness,
        "closure_sha256": closure["closure_sha256"], "predecessors": predecessors,
        "metadata_facts": facts, "path_map_sha256": _path_map_sha256(paths_by_surface),
        "anchor_score_descriptor_sha256": anchor_descriptor_sha,
        "anchor_window_authority_sha256": _canonical_json_sha(anchor_window_authority),
        "anchor_window_authority": anchor_window_authority,
    })


def _issue_live_a0_capability(*, repo_root: Path, root: Path, surface: str,
                              roster: Sequence[str], cpre_terminal_sha256: str) -> _Capability:
    """Root-only issuer that derives A0 inputs from held C-Pre/anchor receipts."""
    _require(surface in plan.SURFACE_ORDER and tuple(roster) == tuple(sorted(roster)) and bool(roster),
             "A0 live issuer surface/roster order drift")
    closure = source_closure(repo_root)
    assert_bound_review(closure)
    cpre = _future_cpre_completion(repo_root=repo_root, expected_terminal_sha256=cpre_terminal_sha256,
                                   closure=closure)
    anchor, anchor_sha = _future_static_anchor(repo_root)
    query_authority = _derive_a0_query_window_authority(
        metadata=cpre["metadata"], anchor_payload=anchor, surface=surface, roster=roster)
    scheduler = scheduler_profile(surface)
    scheduler_attestation = attest_a0_runtime_scheduler(surface=surface)
    witness = _prospective_root_witness(repo_root=repo_root, relative=plan.A0_ROOTS[surface])
    _verify_prospective_root_witness(witness=witness, repo_root=repo_root,
                                     expected_relative=plan.A0_ROOTS[surface], supplied_root=root)
    return _Capability(_ISSUER_TOKEN, {
        "mode": "live", "root_relative": plan.A0_ROOTS[surface], "root_witness": witness,
        "closure_sha256": closure["closure_sha256"], "surface": surface, "roster": list(roster),
        "cpre_terminal_sha256": cpre["terminal_sha256"], "cpre_metadata_inventory_sha256": cpre["metadata_inventory_sha256"],
        "static_anchor_payload": anchor, "static_anchor_descriptor_sha256": anchor_sha,
        "static_anchor_payload_sha256": _canonical_json_sha(anchor),
        "query_window_authority": query_authority,
        "query_window_authority_sha256": _canonical_json_sha(query_authority),
        "device_profile": static_device_profile(), "scheduler_profile": scheduler,
        "scheduler_attestation": scheduler_attestation,
        "frozen_decode_batch_size": plan.A0_FROZEN_DECODE_BATCH_SIZE,
        "resolved_window_size": plan.WINDOW_BINS,
    })


def validate_static_anchor(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen local M10/activity30 anchor receipt body.

    This codec intentionally accepts a supplied mapping for CPU tests.  Live
    admission descriptor-reads the bound score body only after capability;
    importing it cannot access any result root.
    """
    _require(payload.get("schema") == "m2_t4_activity_budget_screen_v1"
             and payload.get("status") == "TERMINAL", "static anchor schema/status drift")
    _require(payload.get("checkpoint_sha256") == plan.ANCHOR_CHECKPOINT_SHA256
             and payload.get("normalization_sha256") == plan.ANCHOR_NORMALIZER_SHA256,
             "static anchor checkpoint/normalizer drift")
    _require(payload.get("parameter_updates") == 0 and payload.get("target_gradients") == 0,
             "static anchor must be inference-only")
    rows = payload.get("rows")
    _require(isinstance(rows, list), "static anchor rows missing")
    matching = [row for row in rows if isinstance(row, Mapping) and row.get("cell") == plan.ANCHOR_CELL]
    _require(matching, "ridge_activity30_m10 anchor cell missing")
    return {
        "anchor_cell": plan.ANCHOR_CELL,
        "checkpoint_sha256": plan.ANCHOR_CHECKPOINT_SHA256,
        "normalization_sha256": plan.ANCHOR_NORMALIZER_SHA256,
        "parameter_updates": 0,
        "target_gradients": 0,
    }


def _exact_absent(parent: Path, name: str) -> None:
    _require("/" not in name and name not in ("", ".", ".."), "root name must be a one-component leaf")
    _require(parent.is_dir() and not parent.is_symlink(), "reservation parent invalid")
    _require(not (parent / name).exists(), "canonical result root must be absent before reservation")


def build_cpre_attempt(*, closure: Mapping[str, Any], metadata_authority: Mapping[str, Any]) -> dict[str, Any]:
    assert_bound_review(closure)
    return {
        "schema": "m2_cpre_attempt_v1",
        "status": "ATTEMPT_RESERVED_BEFORE_METADATA",
        "route": plan.ROUTE_SCHEMA,
        "root_relative": plan.CPRE_ROOT_RELATIVE,
        "source_closure": dict(closure),
        "metadata_authority": dict(metadata_authority),
        "no_model_checkpoint_cuda_r2_or_target_read_before_inventory": True,
        "target_access": False,
    }


def build_a0_attempt(*, surface: str, closure: Mapping[str, Any], cpre_terminal_sha256: str,
                     roster: Sequence[str], batch_size: int = plan.A0_FROZEN_DECODE_BATCH_SIZE,
                     execution_profile: A0ExecutionProfile = V1_EXECUTION_PROFILE) -> dict[str, Any]:
    execution_profile.closure_validator(closure)
    _require(surface in plan.SURFACE_ORDER, "unknown A0 surface")
    _require(len(set(roster)) == len(roster) and bool(roster), "A0 roster must be nonempty and unique")
    _require(int(batch_size) == plan.A0_FROZEN_DECODE_BATCH_SIZE,
             "A0 attempt must bind the frozen decode batch law")
    return {
        "schema": execution_profile.attempt_schema,
        "status": "ATTEMPT_RESERVED_BEFORE_RUNTIME",
        "route": execution_profile.route,
        "surface": surface,
        "root_relative": execution_profile.root_for(surface),
        "source_closure": dict(closure),
        "cpre_terminal_sha256": str(cpre_terminal_sha256),
        "roster": list(roster),
        "device_profile": static_device_profile(),
        "scheduler_profile": scheduler_profile(surface),
        "frozen_decode_batch_size": plan.A0_FROZEN_DECODE_BATCH_SIZE,
        "target_optimization_forbidden": True,
        "parameter_updates": 0,
        "target_gradients": 0,
    }


def _validate_carrier_evidence(roster: Sequence[str],
                               carrier_evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require(set(carrier_evidence) == set(roster), "carrier evidence roster drift")
    canonical: dict[str, Any] = {}
    for session in roster:
        evidence = carrier_evidence[session]
        _require(evidence.get("budget") == plan.CARRIER_BUDGET
                 and evidence.get("selection") == "chronological_first_m"
                 and evidence.get("selected_indices") == list(plan.M10_CHRONOLOGICAL_SELECTED_INDICES)
                 and evidence.get("selected_indices_sha256") == plan.M10_CHRONOLOGICAL_SELECTED_INDICES_SHA256,
                 "A0 carrier must be the anchor's chronological [0..9] M10 authority")
        _require(int(evidence.get("usable_directional_trials", -1)) >= 3,
                 "M10 anchor ridge needs three finite directions within [0..9]")
        _sha256_literal(str(evidence.get("raw_t4_sha256", "")), field="raw_t4_sha256")
        _sha256_literal(str(evidence.get("normalized_t4_sha256", "")), field="normalized_t4_sha256")
        canonical[session] = {
            "budget": plan.CARRIER_BUDGET,
            "selection": "chronological_first_m",
            "selected_indices": list(plan.M10_CHRONOLOGICAL_SELECTED_INDICES),
            "selected_indices_sha256": plan.M10_CHRONOLOGICAL_SELECTED_INDICES_SHA256,
            "usable_directional_trials": int(evidence["usable_directional_trials"]),
            "raw_t4_sha256": str(evidence["raw_t4_sha256"]),
            "normalized_t4_sha256": str(evidence["normalized_t4_sha256"]),
        }
    return canonical


def _validate_activity_seed_evidence(roster: Sequence[str],
                                     seed_evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require(set(seed_evidence) == set(roster), "activity-seed evidence roster drift")
    canonical: dict[str, Any] = {}
    for session in roster:
        evidence = seed_evidence[session]
        _require(evidence.get("seed_source") == "dataset.calib_trialized_neural_features[:30]"
                 and evidence.get("activity_rows") == plan.ACTIVITY_CAPACITY,
                 "A0 seed must be the sealed anchor's direct first30 calibration stack")
        _require(evidence.get("reconstruction_law") == "frozen_cubic_interpolation",
                 "A0 reconstruction audit must use frozen cubic interpolation, never CDM linear rows")
        for field in ("sealed_first30_activity_stack_sha256", "used_seed_activity_stack_sha256",
                      "reconstructed_first30_activity_stack_sha256"):
            _sha256_literal(str(evidence.get(field, "")), field=field)
        _require(evidence.get("used_seed_activity_stack_sha256")
                 == evidence.get("sealed_first30_activity_stack_sha256"),
                 "A0 must seed from sealed direct activity30 bytes, not reconstructed rows")
        max_abs = float(evidence.get("reconstructed_vs_sealed_max_abs", float("inf")))
        _require(np_isfinite(max_abs) and max_abs >= 0.0 and max_abs <= 1e-6,
                 "reconstructed B3S audit exceeds registered <=1e-6 tolerance")
        canonical[session] = {
            "seed_source": "dataset.calib_trialized_neural_features[:30]",
            "reconstruction_law": "frozen_cubic_interpolation",
            "activity_rows": plan.ACTIVITY_CAPACITY,
            "sealed_first30_activity_stack_sha256": str(evidence["sealed_first30_activity_stack_sha256"]),
            "used_seed_activity_stack_sha256": str(evidence["used_seed_activity_stack_sha256"]),
            "reconstructed_first30_activity_stack_sha256": str(evidence["reconstructed_first30_activity_stack_sha256"]),
            "reconstructed_vs_sealed_max_abs": max_abs,
        }
    return canonical


def _validate_query_surface_evidence(surface: str, roster: Sequence[str],
                                     evidence_by_session: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require(set(evidence_by_session) == set(roster), "query-surface evidence roster drift")
    output: dict[str, Any] = {}
    for session in roster:
        evidence = evidence_by_session[session]
        for field in ("governed_post30_cpre_int64_sha256", "governed_post30_anchor_core_sha256",
                      "cpre_anchor_reference_int64_sha256", "anchor_score_ordered_window_starts_sha256"):
            _sha256_literal(str(evidence.get(field, "")), field=field)
        _require(int(evidence.get("governed_post30_window_count", 0)) > 0
                 and int(evidence.get("anchor_reference_window_count", 0)) > 0,
                 "query-surface window count must be positive")
        _require(evidence.get("governed_post30_subset_of_anchor_full_query") is True,
                 "A0 post30 surface must be a deterministic subset of anchor full query")
        if surface == "within_post30":
            _require(evidence.get("within_post30_exact_anchor_parity") is True,
                     "within A0 surface must exactly match its post30 anchor")
        else:
            _require(evidence.get("within_post30_exact_anchor_parity") is None,
                     "external A0 must not mislabel full-query anchor as exact post30 parity")
        batch = evidence.get("batch_decode_evidence")
        _require(isinstance(batch, Mapping) and set(batch) == {"frozen_batch_size", "arms"}
                 and int(batch.get("frozen_batch_size", 0)) > 0,
                 "A0 batch decode receipt schema/law drift")
        arms = batch.get("arms")
        _require(isinstance(arms, Mapping)
                 and list(arms) == ["static", "true_trial", "chunk_phase0", "chunk_phase50"],
                 "A0 batch decode arm order drift")
        canonical_arms: dict[str, Any] = {}
        expected_count = int(evidence["governed_post30_window_count"])
        for arm in ("static", "true_trial", "chunk_phase0", "chunk_phase50"):
            arm_evidence = arms[arm]
            _require(isinstance(arm_evidence, Mapping)
                     and set(arm_evidence) == {"frozen_batch_size", "first_prediction_parity_singleton_batch",
                                                "actual_scored_prediction_count",
                                                "identity_segments", "identity_segments_sha256"}
                     and int(arm_evidence.get("frozen_batch_size", 0)) == int(batch["frozen_batch_size"])
                     and arm_evidence.get("first_prediction_parity_singleton_batch") is True
                     and int(arm_evidence.get("actual_scored_prediction_count", -1)) == expected_count,
                     "A0 batch decode count/law drift")
            segments = arm_evidence.get("identity_segments")
            _require(isinstance(segments, list) and bool(segments), "A0 identity segment receipt missing")
            running = 0
            canonical_segments: list[dict[str, Any]] = []
            for item in segments:
                expected_segment = {"first_ordinal", "last_ordinal", "scored_prediction_count",
                                    "pre_update_identity_sha256", "governed_starts_sha256"}
                _require(isinstance(item, Mapping) and set(item) == expected_segment,
                         "A0 identity segment key schema drift")
                first, last, count = (int(item["first_ordinal"]), int(item["last_ordinal"]),
                                      int(item["scored_prediction_count"]))
                _require(first == running and last >= first and count == last - first + 1,
                         "A0 identity segment does not cover canonical endpoint order")
                _sha256_literal(str(item["pre_update_identity_sha256"]), field="pre_update_identity_sha256")
                _sha256_literal(str(item["governed_starts_sha256"]), field="segment_governed_starts_sha256")
                canonical_segments.append({
                    "first_ordinal": first, "last_ordinal": last, "scored_prediction_count": count,
                    "pre_update_identity_sha256": str(item["pre_update_identity_sha256"]),
                    "governed_starts_sha256": str(item["governed_starts_sha256"]),
                })
                running += count
            _require(running == expected_count, "A0 identity segments do not cover every governed endpoint")
            expected_segments_sha = _canonical_json_sha({"identity_segments": canonical_segments})
            _require(arm_evidence.get("identity_segments_sha256") == expected_segments_sha,
                     "A0 identity segment digest drift")
            canonical_arms[arm] = {
                "frozen_batch_size": int(batch["frozen_batch_size"]),
                "first_prediction_parity_singleton_batch": True,
                "actual_scored_prediction_count": expected_count,
                "identity_segments": canonical_segments,
                "identity_segments_sha256": expected_segments_sha,
            }
        output[session] = {
            "governed_post30_cpre_int64_sha256": str(evidence["governed_post30_cpre_int64_sha256"]),
            "governed_post30_anchor_core_sha256": str(evidence["governed_post30_anchor_core_sha256"]),
            "governed_post30_window_count": int(evidence["governed_post30_window_count"]),
            "cpre_anchor_reference_int64_sha256": str(evidence["cpre_anchor_reference_int64_sha256"]),
            "anchor_score_ordered_window_starts_sha256": str(evidence["anchor_score_ordered_window_starts_sha256"]),
            "anchor_reference_window_count": int(evidence["anchor_reference_window_count"]),
            "governed_post30_subset_of_anchor_full_query": True,
            "within_post30_exact_anchor_parity": evidence.get("within_post30_exact_anchor_parity"),
            "batch_decode_evidence": {
                "frozen_batch_size": int(batch["frozen_batch_size"]),
                "arms": canonical_arms,
            },
        }
    return output


def build_a0_input_authority(*, surface: str, roster: Sequence[str],
                             cpre_terminal_sha256: str, static_anchor: Mapping[str, Any],
                             carrier_evidence: Mapping[str, Mapping[str, Any]],
                             activity_seed_evidence: Mapping[str, Mapping[str, Any]],
                             query_surface_evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The complete frozen M10-carrier/M30-activity contract for one shard."""
    anchor = validate_static_anchor(static_anchor)
    return {
        "schema": "m2_a0_input_authority_v1",
        "surface": surface,
        "roster": list(roster),
        "cpre_terminal_sha256": cpre_terminal_sha256,
        "carrier_label_budget": plan.CARRIER_BUDGET,
        "carrier_support_law": "chronological_first10_positions_in_first30_with_existing_finite_direction_checks",
        "initial_activity_identity": "chronological_first30_b3s_rows",
        "activity_retained_capacity": plan.ACTIVITY_CAPACITY,
        "route_local_state": "SeededActivity30RollingMemory",
        "chunk_input_domain": "raw_contiguous_neural_count_per_20ms_bins",
        "chunk_activity_shape": "[100,N]",
        "chunk_channel_order": "same_raw_neural_channel_order_as_frozen_B3S_pre_pool_authority",
        "chunk_vs_calibration_mechanism_disclosure": (
            "chunk rows are raw fixed-duration 100-bin activity; calibration-visible "
            "B3S rows are existing trial-materialized/interpolated activity rows"),
        "phase_origin": "raw_first30_calibration_boundary_only",
        "first_scored_window_does_not_redefine_chunk_origin": True,
        "behavior_or_angle_labels_read_by_chunk_state": False,
        "metric_mask_used_to_form_or_filter_chunk": False,
        "padding_forbidden": True,
        "anchor": anchor,
        "per_session_anchor_ridge_evidence": _validate_carrier_evidence(roster, carrier_evidence),
        "per_session_activity_seed_evidence": _validate_activity_seed_evidence(roster, activity_seed_evidence),
        "per_session_query_surface_evidence": _validate_query_surface_evidence(
            surface, roster, query_surface_evidence),
        "external_historical_anchor_r2_context_only": surface == "external_post30_local",
        "static_activity30_rescored_on_governed_post30_rows": True,
    }


@dataclass(frozen=True)
class A0Row:
    surface: str
    session_id: str
    arm: str
    phase: int | None
    r2: float
    first_prediction_sha256: str
    prediction_sha256: str
    initial_b3s_identity_sha256: str
    final_b3s_identity_sha256: str
    initial_activity_stack_sha256: str
    final_activity_stack_sha256: str
    query_starts_sha256: str
    target_window_sha256: str
    window_count: int
    update_count: int
    eviction_count: int
    partial_final_bins_discarded: int
    parameter_updates: int
    target_gradients: int
    wall_seconds: float
    peak_memory_bytes: int

    def body(self) -> dict[str, Any]:
        _require(self.arm in ("static", "true_trial", "chunk"), "unknown A0 arm")
        _require(self.arm != "chunk" or self.phase in (0, 50), "chunk row needs phase 0 or 50")
        _require(self.arm == "chunk" or self.phase is None, "only chunks may have a phase")
        _require(np_isfinite(self.r2), "A0 R2 must be finite")
        for field, value in (
            ("first_prediction_sha256", self.first_prediction_sha256),
            ("prediction_sha256", self.prediction_sha256),
            ("initial_b3s_identity_sha256", self.initial_b3s_identity_sha256),
            ("final_b3s_identity_sha256", self.final_b3s_identity_sha256),
            ("initial_activity_stack_sha256", self.initial_activity_stack_sha256),
            ("final_activity_stack_sha256", self.final_activity_stack_sha256),
            ("query_starts_sha256", self.query_starts_sha256),
            ("target_window_sha256", self.target_window_sha256),
        ):
            _sha256_literal(value, field=field)
        _require(self.window_count > 0, "A0 actual scored window count must be positive")
        _require(self.update_count >= 0 and self.eviction_count >= 0
                 and self.partial_final_bins_discarded >= 0, "negative replay counters")
        _require(self.parameter_updates == 0 and self.target_gradients == 0,
                "A0 forbids optimization/target gradients")
        _require(np_isfinite(self.wall_seconds) and self.wall_seconds >= 0.0
                 and self.peak_memory_bytes >= 0, "A0 resource evidence drift")
        return {
            "surface": self.surface, "session_id": self.session_id, "arm": self.arm,
            "phase": self.phase, "r2": float(self.r2),
            "first_prediction_sha256": self.first_prediction_sha256,
            "prediction_sha256": self.prediction_sha256,
            "initial_b3s_identity_sha256": self.initial_b3s_identity_sha256,
            "final_b3s_identity_sha256": self.final_b3s_identity_sha256,
            "initial_activity_stack_sha256": self.initial_activity_stack_sha256,
            "final_activity_stack_sha256": self.final_activity_stack_sha256,
            "query_starts_sha256": self.query_starts_sha256,
            "target_window_sha256": self.target_window_sha256,
            "window_count": int(self.window_count),
            "update_count": int(self.update_count), "eviction_count": int(self.eviction_count),
            "partial_final_bins_discarded": int(self.partial_final_bins_discarded),
            "parameter_updates": 0, "target_gradients": 0,
            "wall_seconds": float(self.wall_seconds), "peak_memory_bytes": int(self.peak_memory_bytes),
        }


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def decode_precomputed_identity_gpu_safe(*, torch: Any, student: Any, neural_windows: Any,
                                         identity: Any, device: Any, batch_size: int,
                                         behavior_scale: float) -> Any:
    """Minimal route-local GPU-safe precomputed-identity decoder adapter.

    The old memory-law helper is CPU-only: it leaves neural batches on CPU and
    calls ``.numpy()`` directly.  A0 must never use it on GPU0.  This adapter
    does not implement a second decoder: it delegates exactly to the frozen
    model's ``decode_with_identity``, transfers only the input batch to the
    already-attested device, then returns CPU float32 predictions for the
    governed local metric.  It is called only after future opaque admission.
    """
    _require(int(batch_size) > 0 and float(behavior_scale) > 0.0,
             "precomputed-identity decode batch/scale drift")
    _require(getattr(neural_windows, "ndim", None) == 3
             and int(neural_windows.shape[1]) == plan.WINDOW_BINS,
             "A0 decoder requires exact W50 neural windows, never B3S [100,N] rows")
    _require(getattr(student, "decoder_mode", None) == "coupled"
             and callable(getattr(student, "decode_with_identity", None)),
             "A0 requires the frozen coupled decode_with_identity contract")
    values: list[Any] = []
    with torch.inference_mode():
        for offset in range(0, int(neural_windows.shape[0]), int(batch_size)):
            source = neural_windows[offset : offset + batch_size]
            neural = torch.from_numpy(source).to(device)
            prediction = student.decode_with_identity(neural, identity)
            _require(bool(torch.isfinite(prediction).all().item()), "A0 decoder output is nonfinite")
            values.append((prediction[:, -1, :].detach().cpu().numpy().astype("float32", copy=False)
                           / float(behavior_scale)))
    import numpy as np
    output = np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32)
    _require(output.shape == (int(neural_windows.shape[0]), 2) and bool(np.isfinite(output).all()),
             "A0 decoded prediction shape/finiteness drift")
    return output


@dataclass(frozen=True)
class IdentityDecodeRequest:
    """One governed endpoint bound to its *pre-update* identity snapshot.

    The causal raw-bin transaction is intentionally separate from this record:
    it is emitted only for a governed endpoint, while its identity is sampled
    before the transaction's possible true-trial/chunk update.  Consecutive
    records with the same identity can therefore be safely decoded in one
    batch without changing either state chronology or prediction order.
    """

    ordinal: int
    raw_window_start: int
    identity: Any
    identity_sha256: str


def decode_identity_snapshot_segments(*, torch: Any, student: Any, neural_source: Any,
                                      requests: Sequence[IdentityDecodeRequest], device: Any,
                                      batch_size: int, behavior_scale: float,
                                      array_sha256: Callable[[Any], str],
                                      force_first_singleton: bool = True) -> tuple[Any, dict[str, Any]]:
    """Decode chronological endpoint snapshots in identity-homogeneous batches.

    This is deliberately an adapter around :func:`decode_precomputed_identity_gpu_safe`,
    not a second decoder.  The request order is canonical governed-window order;
    output is restored by ordinal after every bounded batch.  The segment witness
    commits to the identity that governed every endpoint before its update.
    """
    import numpy as np

    _require(int(batch_size) > 0, "A0 frozen decode batch size must be positive")
    ordered = tuple(requests)
    _require(bool(ordered), "A0 needs at least one governed decode request")
    _require([request.ordinal for request in ordered] == list(range(len(ordered))),
             "A0 decode requests must be canonical contiguous ordinals")
    starts = [request.raw_window_start for request in ordered]
    _require(all(isinstance(start, int) for start in starts)
             and all(right > left for left, right in zip(starts, starts[1:])),
             "A0 decode requests must have strictly ordered raw starts")
    for request in ordered:
        _sha256_literal(request.identity_sha256, field="pre_update_identity_sha256")

    neural_matrix = np.ascontiguousarray(np.asarray(neural_source, dtype=np.float32))
    _require(neural_matrix.ndim == 2, "A0 raw neural source must be [T,N]")
    output: list[Any | None] = [None] * len(ordered)
    segments: list[dict[str, Any]] = []
    segment_start = 0
    while segment_start < len(ordered):
        segment_end = segment_start + 1
        snapshot = ordered[segment_start]
        while (segment_end < len(ordered)
               and not (force_first_singleton and segment_start == 0)
               and ordered[segment_end].identity_sha256 == snapshot.identity_sha256):
            segment_end += 1
        current = ordered[segment_start:segment_end]
        # A SHA collision must never silently bind two different in-memory
        # snapshots in a segment.  This check is cheap (one identity per
        # governed segment), and retains a strict pre-update identity law.
        for request in current[1:]:
            _require(bool(torch.equal(snapshot.identity, request.identity)),
                     "same A0 identity digest mapped to unequal snapshots")
        windows = np.ascontiguousarray(np.stack([
            neural_matrix[request.raw_window_start:request.raw_window_start + plan.WINDOW_BINS]
            for request in current
        ], axis=0), dtype=np.float32)
        _require(windows.shape == (len(current), plan.WINDOW_BINS, neural_matrix.shape[1]),
                 "A0 governed W50 window/source topology drift")
        values = decode_precomputed_identity_gpu_safe(
            torch=torch, student=student, neural_windows=windows, identity=snapshot.identity,
            device=device, batch_size=int(batch_size), behavior_scale=float(behavior_scale))
        _require(values.shape == (len(current), 2), "A0 segment prediction topology drift")
        for request, value in zip(current, values, strict=True):
            output[request.ordinal] = np.ascontiguousarray(value, dtype=np.float32)
        segment_starts = np.ascontiguousarray([request.raw_window_start for request in current], dtype=np.int64)
        segments.append({
            "first_ordinal": int(current[0].ordinal),
            "last_ordinal": int(current[-1].ordinal),
            "scored_prediction_count": int(len(current)),
            "pre_update_identity_sha256": snapshot.identity_sha256,
            "governed_starts_sha256": str(array_sha256(segment_starts)),
        })
        segment_start = segment_end
    _require(all(value is not None for value in output), "A0 batch decode left an endpoint unresolved")
    predictions = np.ascontiguousarray(np.stack([value for value in output if value is not None], axis=0),
                                       dtype=np.float32)
    _require(predictions.shape == (len(ordered), 2) and bool(np.isfinite(predictions).all()),
             "A0 batched prediction output drift")
    evidence = {
        "frozen_batch_size": int(batch_size),
        "first_prediction_parity_singleton_batch": bool(force_first_singleton),
        "actual_scored_prediction_count": int(len(ordered)),
        "identity_segments": segments,
        "identity_segments_sha256": _canonical_json_sha({"identity_segments": segments}),
    }
    return predictions, evidence


def validate_a0_rows(rows: Iterable[A0Row], *, surface: str, roster: Sequence[str]) -> dict[str, Any]:
    ordered = list(rows)
    expected_keys = [(session, arm, phase) for session in roster
                     for arm, phase in (("static", None), ("true_trial", None), ("chunk", 0), ("chunk", 50))]
    _require([(row.session_id, row.arm, row.phase) for row in ordered] == expected_keys,
             "A0 rows must be canonical roster/static/true/chunk0/chunk50 order")
    bodies = [row.body() for row in ordered]
    for session in roster:
        group = [row for row in ordered if row.session_id == session]
        initial = {row.initial_b3s_identity_sha256 for row in group}
        initial_stack = {row.initial_activity_stack_sha256 for row in group}
        first_prediction = {row.first_prediction_sha256 for row in group}
        query = {row.query_starts_sha256 for row in group}
        target = {row.target_window_sha256 for row in group}
        _require(len(initial) == len(initial_stack) == len(first_prediction) == len(query) == len(target) == 1,
                 "static/true/chunks must share first B3S/activity-stack/input/prediction parity")
    true = {row.session_id: row.r2 for row in ordered if row.arm == "true_trial"}
    phase0 = {row.session_id: row.r2 for row in ordered if row.arm == "chunk" and row.phase == 0}
    deltas = {session: float(phase0[session] - true[session]) for session in roster}
    values = list(deltas.values())
    mean = sum(values) / len(values)
    worst = min(values)
    if surface == "external_post30_local":
        gate = {
            "governing_surface": True,
            "external_noninferiority_mean": mean >= plan.EXTERNAL_GATE_MEAN,
            "external_noninferiority_worst": worst >= plan.EXTERNAL_GATE_WORST,
            "materially_better": mean > plan.MATERIAL_BETTER_MEAN and sum(value > 0.0 for value in values) >= 4,
        }
    else:
        gate = {
            "governing_surface": False,
            "disclosure": "within_post30_secondary_not_a_governing_gate",
            "phase0_minus_true_mean": mean,
            "phase0_minus_true_worst": worst,
        }
    return {
        "rows": bodies,
        "row_count": len(bodies),
        "phase0_minus_true": {key: deltas[key] for key in roster},
        "phase0_minus_true_mean": mean,
        "phase0_minus_true_worst": worst,
        "gate": gate,
        "rows_sha256": _canonical_json_sha({"rows": bodies}),
    }


def build_a0_replay(*, surface: str, roster: Sequence[str], rows: Iterable[A0Row],
                    input_authority_sha256: str, actual_scored_prediction_count: Mapping[str, int]) -> dict[str, Any]:
    """Recompute the row/gate body independently of a runtime callback."""
    summary = validate_a0_rows(rows, surface=surface, roster=roster)
    _require(set(actual_scored_prediction_count) == set(roster)
             and all(int(value) > 0 for value in actual_scored_prediction_count.values()),
             "actual governed prediction count must be positive per roster session")
    return {
        "schema": "m2_a0_replay_v1",
        "surface": surface,
        "input_authority_sha256": input_authority_sha256,
        "actual_scored_prediction_count": {key: int(actual_scored_prediction_count[key]) for key in roster},
        "summary": summary,
        "parameter_updates": 0,
        "target_gradients": 0,
    }


def build_a0_terminal(*, attempt_sha256: str, launch_sha256: str, input_authority_sha256: str,
                      replay_sha256: str, replay: Mapping[str, Any], closure: Mapping[str, Any],
                      execution_profile: A0ExecutionProfile = V1_EXECUTION_PROFILE,
                      receipt_bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    summary = replay.get("summary")
    _require(isinstance(summary, Mapping) and int(summary.get("row_count", -1)) > 0,
             "cannot terminalize invalid A0 replay")
    payload = {
        "schema": execution_profile.terminal_schema, "status": "TERMINAL",
        "attempt_sha256": attempt_sha256, "launch_sha256": launch_sha256,
        "input_authority_sha256": input_authority_sha256, "replay_sha256": replay_sha256,
        "rows_sha256": summary.get("rows_sha256"), "gate": summary.get("gate"),
        "source_closure": dict(closure), "parameter_updates": 0, "target_gradients": 0,
        "target_optimization": False,
    }
    if receipt_bindings is not None:
        payload["execution_profile_bindings"] = dict(receipt_bindings)
    return payload


def build_aggregate_score(*, external_replay: Mapping[str, Any], within_replay: Mapping[str, Any]) -> dict[str, Any]:
    """CPU-only aggregate recomposition; no callback-provided gate is trusted."""
    _require(external_replay.get("surface") == "external_post30_local"
             and within_replay.get("surface") == "within_post30", "aggregate surface drift")
    external_summary = external_replay.get("summary")
    within_summary = within_replay.get("summary")
    _require(isinstance(external_summary, Mapping) and isinstance(within_summary, Mapping),
             "aggregate replay summaries missing")
    return {
        "schema": "m2_a0_aggregate_score_v1",
        "external_rows_sha256": external_summary.get("rows_sha256"),
        "within_rows_sha256": within_summary.get("rows_sha256"),
        "external_phase0_minus_true": external_summary.get("phase0_minus_true"),
        "within_phase0_minus_true": within_summary.get("phase0_minus_true"),
        "external_gate": external_summary.get("gate"),
        "within_secondary_gate": within_summary.get("gate"),
        "target_optimization": False,
    }


def _row_from_body(body: Mapping[str, Any]) -> A0Row:
    expected = {
        "surface", "session_id", "arm", "phase", "r2", "first_prediction_sha256", "prediction_sha256",
        "initial_b3s_identity_sha256", "final_b3s_identity_sha256", "initial_activity_stack_sha256",
        "final_activity_stack_sha256", "query_starts_sha256", "target_window_sha256", "window_count",
        "update_count", "eviction_count", "partial_final_bins_discarded", "parameter_updates",
        "target_gradients", "wall_seconds", "peak_memory_bytes",
    }
    _require(set(body) == expected, "serialized A0 row key schema drift")
    return A0Row(**dict(body))


def _completed_a0_shard_witness(*, repo_root: Path, surface: str,
                                closure: Mapping[str, Any]) -> dict[str, str]:
    """Descriptor-only witness for one completed canonical A0 shard."""
    _require(surface in plan.SURFACE_ORDER, "aggregate shard surface drift")
    root = _canonical_result_path(repo_root, plan.A0_ROOTS[surface])
    descriptors = receipts.verify_terminal_xor(root, success_bodies=(
        "attempt.json", "launch.json", "input_authority.json", "replay.json", "terminal.json"))
    _require(set(descriptors) == {"attempt.json", "launch.json", "input_authority.json", "replay.json", "terminal.json"},
             "aggregate shard must be terminal-only")
    terminal = descriptors["terminal.json"].payload
    _require(terminal.get("schema") == "m2_a0_terminal_v1" and terminal.get("status") == "TERMINAL"
             and terminal.get("replay_sha256") == descriptors["replay.json"].sha256
             and terminal.get("source_closure") == dict(closure),
             "aggregate shard terminal/current closure drift")
    return {"root_relative": plan.A0_ROOTS[surface],
            "terminal_sha256": descriptors["terminal.json"].sha256,
            "replay_sha256": descriptors["replay.json"].sha256}


def _issue_live_aggregate_capability(*, repo_root: Path, root: Path) -> _Capability:
    """Root-only CPU admission; no Torch/CUDA/data factory is reachable here."""
    closure = source_closure(repo_root)
    assert_bound_review(closure)
    external = _completed_a0_shard_witness(repo_root=repo_root, surface="external_post30_local", closure=closure)
    within = _completed_a0_shard_witness(repo_root=repo_root, surface="within_post30", closure=closure)
    witness = _prospective_root_witness(repo_root=repo_root, relative=plan.A0_ROOTS["aggregate"])
    _verify_prospective_root_witness(witness=witness, repo_root=repo_root,
                                     expected_relative=plan.A0_ROOTS["aggregate"], supplied_root=root)
    return _Capability(_ISSUER_TOKEN, {
        "mode": "live", "root_relative": plan.A0_ROOTS["aggregate"], "root_witness": witness,
        "closure_sha256": closure["closure_sha256"], "cpu_only": True,
        "external_shard": external, "within_shard": within,
    })


def execute_a0_aggregate_cpu(*, root: Path, repo_root: Path, capability: object,
                             external_root: Path, within_root: Path) -> dict[str, str]:
    """Descriptor-only CPU aggregate; it never imports Torch/CUDA/data."""
    closure = source_closure(repo_root)
    binding = _consume(capability, expected_root=plan.A0_ROOTS["aggregate"], closure=closure,
                       repo_root=repo_root, supplied_root=root)
    if binding.get("mode") == "live":
        _require(binding.get("cpu_only") is True,
                 "aggregate live capability must be explicitly CPU-only")
        _require(external_root.absolute() == _canonical_result_path(
                    repo_root, plan.A0_ROOTS["external_post30_local"]).absolute()
                 and within_root.absolute() == _canonical_result_path(
                    repo_root, plan.A0_ROOTS["within_post30"]).absolute(),
                 "aggregate live executor received noncanonical shard root")
        _require(binding.get("external_shard") == _completed_a0_shard_witness(
                    repo_root=repo_root, surface="external_post30_local", closure=closure)
                 and binding.get("within_shard") == _completed_a0_shard_witness(
                    repo_root=repo_root, surface="within_post30", closure=closure),
                 "aggregate live shard witness drift")
    _exact_absent(root.parent, root.name)
    root.mkdir(mode=0o700)
    reservation = _capture_reserved_root(root) if (binding.get("mode") == "live"
                                                    or execution_profile.final_validator is not None) else None
    if reservation is not None and isinstance(binding, dict):
        binding["_reserved_root"] = dict(reservation)
    attempt = receipts.publish_pair(root, "attempt.json", {
        "schema": "m2_a0_aggregate_attempt_v1", "status": "ATTEMPT_RESERVED_BEFORE_DESCRIPTOR_READ",
        "source_closure": closure, "target_access": False, "cuda_initialized": False,
    })
    stage = "descriptor_read"
    try:
        external = receipts.verify_terminal_xor(external_root, success_bodies=(
            "attempt.json", "launch.json", "input_authority.json", "replay.json", "terminal.json"))
        within = receipts.verify_terminal_xor(within_root, success_bodies=(
            "attempt.json", "launch.json", "input_authority.json", "replay.json", "terminal.json"))
        _require("terminal.json" in external and "terminal.json" in within, "aggregate requires completed A0 shards")
        external_replay = external["replay.json"].payload
        within_replay = within["replay.json"].payload
        for descriptor, replay in ((external, external_replay), (within, within_replay)):
            terminal = descriptor["terminal.json"].payload
            _require(terminal.get("replay_sha256") == descriptor["replay.json"].sha256,
                     "aggregate shard terminal/replay link drift")
            rows = [_row_from_body(row) for row in replay.get("summary", {}).get("rows", [])]
            rebuilt = validate_a0_rows(rows, surface=str(replay.get("surface")),
                                       roster=tuple(dict.fromkeys(row.session_id for row in rows)))
            _require(rebuilt["rows_sha256"] == replay.get("summary", {}).get("rows_sha256"),
                     "aggregate independent shard replay recomposition drift")
        score = build_aggregate_score(external_replay=external_replay, within_replay=within_replay)
        stage = "input_authority"
        inputs = receipts.publish_pair(root, "input_authority.json", {
            "schema": "m2_a0_aggregate_input_authority_v1", "external_terminal_sha256": external["terminal.json"].sha256,
            "within_terminal_sha256": within["terminal.json"].sha256,
        })
        stage = "score"
        score_descriptor = receipts.publish_pair(root, "score.json", score)
        stage = "terminal"
        if reservation is not None:
            _verify_reserved_root(root, reservation)
        terminal = receipts.publish_pair(root, "terminal.json", {
            "schema": "m2_a0_aggregate_terminal_v1", "status": "TERMINAL", "attempt_sha256": attempt.sha256,
            "input_authority_sha256": inputs.sha256, "score_sha256": score_descriptor.sha256,
            "source_closure": closure, "cuda_initialized": False,
        })
        receipts.verify_terminal_xor(root, success_bodies=("attempt.json", "input_authority.json", "score.json", "terminal.json"))
        return {"attempt": attempt.sha256, "input_authority": inputs.sha256,
                "score": score_descriptor.sha256, "terminal": terminal.sha256}
    except Exception as exc:
        names = set(os.listdir(root))
        prefix = tuple(name for name in ("attempt.json", "input_authority.json", "score.json") if name in names)
        try:
            receipts.publish_failure_preserving_prefix(
                root=root, prefix_bodies=prefix, attempt_sha256=attempt.sha256,
                schema="m2_a0_aggregate_failure_v1", stage=stage,
                error=f"{type(exc).__name__}: {exc}",
                progress={"target_access": False, "cuda_initialized": False, "descriptor_read_attempted": True},
            )
        except Exception as failure_exc:
            raise PhysicalContractError("A0 aggregate failure could not retain its immutable prefix") from failure_exc
        raise PhysicalContractError("A0 aggregate failed after immutable attempt") from exc


def execute_a0_shard_synthetic(*, root: Path, repo_root: Path, capability: object,
                               surface: str, roster: Sequence[str], cpre_terminal_sha256: str,
                               static_anchor: Mapping[str, Any], rows: Iterable[A0Row],
                               actual_scored_prediction_count: Mapping[str, int],
                               carrier_evidence: Mapping[str, Mapping[str, Any]],
                               activity_seed_evidence: Mapping[str, Mapping[str, Any]],
                               query_surface_evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Typed no-data lifecycle seam for root audit tests.

    The supplied A0 rows are synthetic evidence, not a model callback.  The
    function is intentionally incapable of opening data/checkpoints or CUDA;
    its role is to establish attempt/order/link/topology contracts before a
    later reviewed physical factory is connected.
    """
    _require(surface in plan.SURFACE_ORDER and root.name == Path(plan.A0_ROOTS[surface]).name,
             "A0 exact named root/surface drift")
    closure = source_closure(repo_root)
    _consume_a0(capability, surface=surface, closure=closure,
                cpre_terminal_sha256=cpre_terminal_sha256, roster=roster)
    _exact_absent(root.parent, root.name)
    root.mkdir(mode=0o700)
    attempt = receipts.publish_pair(root, "attempt.json", build_a0_attempt(
        surface=surface, closure=closure, cpre_terminal_sha256=cpre_terminal_sha256, roster=roster,
    ))
    launch = receipts.publish_pair(root, "launch.json", {
        "schema": "m2_a0_launch_v1", "attempt_sha256": attempt.sha256,
        "device_profile": static_device_profile(), "scheduler_profile": scheduler_profile(surface),
        "frozen_decode_batch_size": plan.A0_FROZEN_DECODE_BATCH_SIZE,
        "parameter_updates": 0, "target_gradients": 0,
    })
    authority = build_a0_input_authority(surface=surface, roster=roster,
                                         cpre_terminal_sha256=cpre_terminal_sha256,
                                         static_anchor=static_anchor, carrier_evidence=carrier_evidence,
                                         activity_seed_evidence=activity_seed_evidence,
                                         query_surface_evidence=query_surface_evidence)
    input_descriptor = receipts.publish_pair(root, "input_authority.json", authority)
    replay = build_a0_replay(surface=surface, roster=roster, rows=rows,
                             input_authority_sha256=input_descriptor.sha256,
                             actual_scored_prediction_count=actual_scored_prediction_count)
    replay_descriptor = receipts.publish_pair(root, "replay.json", replay)
    terminal = receipts.publish_pair(root, "terminal.json", build_a0_terminal(
        attempt_sha256=attempt.sha256, launch_sha256=launch.sha256,
        input_authority_sha256=input_descriptor.sha256, replay_sha256=replay_descriptor.sha256,
        replay=replay, closure=closure,
    ))
    descriptors = receipts.verify_topology(root, bodies=(
        "attempt.json", "launch.json", "input_authority.json", "replay.json", "terminal.json"))
    _require(descriptors["terminal.json"].payload.get("attempt_sha256") == attempt.sha256
             and descriptors["terminal.json"].payload.get("replay_sha256") == replay_descriptor.sha256,
             "A0 final descriptor-link drift")
    return {key.removesuffix(".json"): value.sha256 for key, value in descriptors.items()}


def _publish_cpre_failure_after_attempt(*, root: Path, attempt_sha256: str, stage: str,
                                        error: str, progress: Mapping[str, Any],
                                        failure_schema: str = "m2_cpre_failure_v1") -> None:
    """Publish the honest C-Pre failure topology for its completed stage.

    A complete metadata pair is immutable evidence and is retained.  A body or
    sidecar without its mate is deliberately not papered over with a failure:
    that is an interrupted pair publication requiring root audit.
    """
    names = set(os.listdir(root))
    _require("attempt.json" in names and "attempt.json.sha256" in names,
             "C-Pre immutable attempt pair absent before failure publication")
    metadata_pair = {"metadata_inventory.json", "metadata_inventory.json.sha256"}
    present = metadata_pair & names
    _require(not present or present == metadata_pair,
             "C-Pre metadata pair is half-published; stop for root audit")
    _require("terminal.json" not in names and "terminal.json.sha256" not in names,
             "C-Pre terminal publication started; stop for root audit")
    prefix = ("attempt.json", "metadata_inventory.json") if present else ("attempt.json",)
    receipts.publish_failure_preserving_prefix(
        root=root, prefix_bodies=prefix, attempt_sha256=attempt_sha256,
        schema=failure_schema, stage=stage, error=error, progress=progress,
    )
    verified = receipts.verify_terminal_xor(root, success_bodies=(
        "attempt.json", "metadata_inventory.json", "terminal.json"))
    _require(tuple(name for name in ("attempt.json", "metadata_inventory.json") if name in verified)
             == prefix, "C-Pre failure retained-prefix topology drift")


def execute_cpre_synthetic(*, root: Path, repo_root: Path, capability: object,
                            records: Sequence[inventory.ChronologyMetadata],
                            fail_after_metadata_for_test: bool = False) -> dict[str, Any]:
    """CPU-only test seam that proves attempt-before-metadata lifecycle.

    It never loads data—the supplied records are pre-built chronology metadata.
    Production admission will call an equivalent metadata-only factory later.
    """
    closure = source_closure(repo_root)
    _consume(capability, expected_root=plan.CPRE_ROOT_RELATIVE, closure=closure)
    _require(root.name == "m2_cpre_metadata_v1", "C-Pre root basename drift")
    _exact_absent(root.parent, root.name)
    root.mkdir(mode=0o700)
    attempt = receipts.publish_pair(root, "attempt.json", build_cpre_attempt(
        closure=closure, metadata_authority={"factory": "synthetic_chronology_only", "target_values_read": False},
    ))
    stage = "metadata_inventory"
    try:
        metadata = inventory.inventory_dataset(records)
        inventory_descriptor = receipts.publish_pair(root, "metadata_inventory.json", metadata)
        stage = "terminal"
        if fail_after_metadata_for_test:
            raise RuntimeError("synthetic requested post-metadata failure")
        terminal_payload = {
            "schema": "m2_cpre_terminal_v1", "status": "TERMINAL",
            "attempt_sha256": attempt.sha256, "metadata_inventory_sha256": inventory_descriptor.sha256,
            "target_access": False, "model_or_checkpoint_opened": False,
            "cuda_initialized": False, "source_closure": closure,
        }
        terminal = receipts.publish_pair(root, "terminal.json", terminal_payload)
        receipts.verify_topology(root, bodies=("attempt.json", "metadata_inventory.json", "terminal.json"))
        return {"attempt": attempt.sha256, "inventory": inventory_descriptor.sha256, "terminal": terminal.sha256}
    except Exception as exc:
        try:
            _publish_cpre_failure_after_attempt(
                root=root, attempt_sha256=attempt.sha256, stage=stage,
                error=f"{type(exc).__name__}: {exc}",
                progress={"target_access": False, "target_values_read": False,
                          "model_or_checkpoint_opened": False, "cuda_initialized": False,
                          "metadata_inventory_published": "metadata_inventory.json" in os.listdir(root)},
            )
        except Exception as failure_exc:
            raise PhysicalContractError("C-Pre synthetic failure could not publish an honest prefix receipt") from failure_exc
        raise PhysicalContractError("C-Pre synthetic lifecycle failed after immutable attempt") from exc


def execute_cpre_metadata_only(*, root: Path, repo_root: Path, capability: object,
                               paths_by_surface: Mapping[str, Sequence[Path]],
                               resolved_metadata_facts: Mapping[str, object],
                               anchor_window_authority: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
                               session_name: Callable[[Path], str] | None = None) -> dict[str, str]:
    """Production C-Pre entry: metadata-only NWB inventory after attempt.

    No full Falcon loader appears in this function.  The provider's public
    interface makes model/checkpoint, finger-velocity values, units/spikes and
    Torch structurally unavailable.  A future root-issued capability must bind
    the metadata facts and per-surface anchor window authority before calling.
    """
    closure = source_closure(repo_root)
    binding = _consume(capability, expected_root=plan.CPRE_ROOT_RELATIVE, closure=closure,
                       repo_root=repo_root, supplied_root=root)
    facts = inventory.validate_resolved_metadata_facts(resolved_metadata_facts)
    if binding.get("mode") == "live":
        current_anchor, current_anchor_sha = _future_static_anchor(repo_root)
        held_authority = binding.get("anchor_window_authority")
        _require(current_anchor_sha == binding.get("anchor_score_descriptor_sha256")
                 and isinstance(held_authority, Mapping)
                 and _canonical_json_sha(_future_anchor_window_authority(anchor_payload=current_anchor))
                 == binding.get("anchor_window_authority_sha256"),
                 "C-Pre live exact anchor-score authority drift")
        _require(anchor_window_authority is None or _canonical_json_sha(anchor_window_authority)
                 == binding.get("anchor_window_authority_sha256"),
                 "caller may not replace root-issued C-Pre anchor authority")
        anchor_window_authority = held_authority
    _require(anchor_window_authority is not None, "C-Pre needs root-issued anchor window authority")
    _require(binding.get("metadata_facts") == facts
             and binding.get("anchor_window_authority_sha256")
             == _canonical_json_sha(anchor_window_authority),
             "C-Pre capability metadata/anchor authority drift")
    if binding.get("mode") == "live":
        _require(binding.get("path_map_sha256") == _path_map_sha256(paths_by_surface),
                 "C-Pre capability path-map/roster drift")
        _require(binding.get("predecessors") == _future_predecessors(repo_root),
                 "C-Pre live predecessor witness drift")
    _require(set(paths_by_surface) == set(plan.SURFACE_ORDER)
             and set(anchor_window_authority) == set(plan.SURFACE_ORDER),
             "C-Pre surface authority topology drift")
    _require(root.name == "m2_cpre_metadata_v1", "C-Pre root basename drift")
    _exact_absent(root.parent, root.name)
    root.mkdir(mode=0o700)
    reservation = _capture_reserved_root(root) if binding.get("mode") == "live" else None
    attempt = receipts.publish_pair(root, "attempt.json", build_cpre_attempt(
        closure=closure,
        metadata_authority={
            "provider": "direct_metadata_only_nwb_timestamp_evalmask_trialstart_v1",
            "resolved_metadata_facts": facts,
            "anchor_window_authority_sha256": _canonical_json_sha(anchor_window_authority),
            "target_values_read": False, "model_or_checkpoint_opened": False, "cuda_initialized": False,
        },
    ))
    stage = "metadata_provider"
    try:
        records: list[inventory.ChronologyMetadata] = []
        evidence: dict[str, Any] = {}
        for surface in plan.SURFACE_ORDER:
            current = inventory.metadata_only_nwb_records(
                paths=paths_by_surface[surface], surface=surface,
                resolved_metadata_facts=facts,
                anchor_window_authority=anchor_window_authority[surface], session_name=session_name,
            )
            records.extend(current)
            evidence[surface] = inventory.crosscheck_anchor_window_authority(
                records=current, surface=surface, authority_by_session=anchor_window_authority[surface])
        metadata = inventory.inventory_dataset(records)
        metadata["metadata_provider"] = "direct_metadata_only_nwb_timestamp_evalmask_trialstart_v1"
        metadata["resolved_metadata_facts"] = facts
        metadata["anchor_window_crosschecks"] = evidence
        metadata["model_or_checkpoint_opened"] = False
        metadata["cuda_initialized"] = False
        descriptor = receipts.publish_pair(root, "metadata_inventory.json", metadata)
        stage = "terminal"
        if reservation is not None:
            _verify_reserved_root(root, reservation)
        terminal = receipts.publish_pair(root, "terminal.json", {
            "schema": "m2_cpre_terminal_v1", "status": "TERMINAL",
            "attempt_sha256": attempt.sha256, "metadata_inventory_sha256": descriptor.sha256,
            "target_access": False, "target_values_read": False,
            "model_or_checkpoint_opened": False, "cuda_initialized": False,
            "source_closure": closure,
        })
        receipts.verify_topology(root, bodies=("attempt.json", "metadata_inventory.json", "terminal.json"))
        return {"attempt": attempt.sha256, "inventory": descriptor.sha256, "terminal": terminal.sha256}
    except Exception as exc:
        try:
            _publish_cpre_failure_after_attempt(
                root=root, attempt_sha256=attempt.sha256, stage=stage,
                error=f"{type(exc).__name__}: {exc}",
                progress={"target_access": False, "target_values_read": False,
                          "model_or_checkpoint_opened": False, "cuda_initialized": False,
                          "metadata_inventory_published": "metadata_inventory.json" in os.listdir(root)},
            )
        except Exception as failure_exc:
            raise PhysicalContractError("C-Pre metadata failure could not publish an honest prefix receipt") from failure_exc
        raise PhysicalContractError("C-Pre metadata-only production lifecycle failed after immutable attempt") from exc


def build_cpre_v2_attempt(*, closure: Mapping[str, Any], metadata_authority: Mapping[str, Any],
                          v1_failure_predecessor: Mapping[str, Any]) -> dict[str, Any]:
    """V2 successor attempt: immutable V1 failure is admission evidence, not retry state."""
    assert_bound_cpre_v2_repair(closure)
    _require(v1_failure_predecessor == {
        "root_relative": plan.CPRE_ROOT_RELATIVE,
        "attempt_sha256": plan.CPRE_V1_FAILURE_ATTEMPT_SHA256,
        "failure_sha256": plan.CPRE_V1_FAILURE_SHA256,
    }, "V2 C-Pre exact V1 failure predecessor drift")
    return {
        "schema": "m2_cpre_attempt_v2", "status": "ATTEMPT_RESERVED_BEFORE_METADATA",
        "route": plan.ROUTE_SCHEMA, "root_relative": plan.CPRE_V2_ROOT_RELATIVE,
        "source_closure": dict(closure), "parent_workorder_sha256": plan.WORKORDER_SHA256,
        "v2_repair_workorder_sha256": plan.CPRE_V2_REPAIR_SHA256,
        "v1_failure_predecessor": dict(v1_failure_predecessor),
        "metadata_authority": dict(metadata_authority),
        "window_coordinate_law": "endpoint_s_ge_raw_trial_start_n_plus_49",
        "no_model_checkpoint_cuda_r2_or_target_read_before_inventory": True,
        "target_access": False,
    }


def _issue_live_cpre_v2_capability(*, repo_root: Path, root: Path,
                                   paths_by_surface: Mapping[str, Sequence[Path]],
                                   resolved_metadata_facts: Mapping[str, object]) -> _Capability:
    """Root-only issuer.  It validates V1's four-leaf failure before V2 reservation."""
    closure = source_closure(repo_root)
    assert_bound_cpre_v2_repair(closure)
    predecessors = _future_predecessors(repo_root)
    v1_failure = _future_v1_cpre_failure_predecessor(repo_root)
    facts = inventory.validate_resolved_metadata_facts(resolved_metadata_facts)
    anchor_payload, anchor_descriptor_sha = _future_static_anchor(repo_root)
    anchor_window_authority = _future_anchor_window_authority(anchor_payload=anchor_payload)
    witness = _prospective_root_witness(repo_root=repo_root, relative=plan.CPRE_V2_ROOT_RELATIVE)
    _verify_prospective_root_witness(witness=witness, repo_root=repo_root,
                                     expected_relative=plan.CPRE_V2_ROOT_RELATIVE, supplied_root=root)
    return _Capability(_ISSUER_TOKEN, {
        "mode": "live", "root_relative": plan.CPRE_V2_ROOT_RELATIVE, "root_witness": witness,
        "closure_sha256": closure["closure_sha256"], "predecessors": predecessors,
        "v1_failure_predecessor": v1_failure, "metadata_facts": facts,
        "path_map_sha256": _path_map_sha256(paths_by_surface),
        "anchor_score_descriptor_sha256": anchor_descriptor_sha,
        "anchor_window_authority_sha256": _canonical_json_sha(anchor_window_authority),
        "anchor_window_authority": anchor_window_authority,
    })


def execute_cpre_v2_synthetic(*, root: Path, repo_root: Path, capability: object,
                              records: Sequence[inventory.ChronologyMetadata],
                              v1_failure_predecessor: Mapping[str, Any],
                              fail_after_metadata_for_test: bool = False) -> dict[str, str]:
    """No-data V2 lifecycle seam; production admission uses the same codec below."""
    closure = source_closure(repo_root)
    assert_bound_cpre_v2_repair(closure)
    binding = _consume(capability, expected_root=plan.CPRE_V2_ROOT_RELATIVE, closure=closure)
    _require(binding.get("v1_failure_predecessor") == dict(v1_failure_predecessor),
             "V2 synthetic capability/predecessor drift")
    _require(root.name == Path(plan.CPRE_V2_ROOT_RELATIVE).name, "V2 C-Pre root basename drift")
    _exact_absent(root.parent, root.name); root.mkdir(mode=0o700)
    attempt = receipts.publish_pair(root, "attempt.json", build_cpre_v2_attempt(
        closure=closure, v1_failure_predecessor=v1_failure_predecessor,
        metadata_authority={"factory": "synthetic_chronology_only", "target_values_read": False},
    ))
    stage = "metadata_inventory"
    try:
        metadata = inventory.inventory_dataset_v2(records)
        inventory_descriptor = receipts.publish_pair(root, "metadata_inventory.json", metadata)
        stage = "terminal"
        if fail_after_metadata_for_test:
            raise RuntimeError("synthetic requested post-metadata failure")
        terminal = receipts.publish_pair(root, "terminal.json", {
            "schema": "m2_cpre_terminal_v2", "status": "TERMINAL",
            "attempt_sha256": attempt.sha256, "metadata_inventory_sha256": inventory_descriptor.sha256,
            "v1_failure_predecessor": dict(v1_failure_predecessor),
            "target_access": False, "target_values_read": False,
            "model_or_checkpoint_opened": False, "cuda_initialized": False,
            "source_closure": dict(closure),
        })
        receipts.verify_topology(root, bodies=("attempt.json", "metadata_inventory.json", "terminal.json"))
        return {"attempt": attempt.sha256, "inventory": inventory_descriptor.sha256, "terminal": terminal.sha256}
    except Exception as exc:
        _publish_cpre_failure_after_attempt(
            root=root, attempt_sha256=attempt.sha256, stage=stage,
            error=f"{type(exc).__name__}: {exc}", failure_schema="m2_cpre_failure_v2",
            progress={"target_access": False, "target_values_read": False,
                      "model_or_checkpoint_opened": False, "cuda_initialized": False,
                      "metadata_inventory_published": "metadata_inventory.json" in os.listdir(root),
                      "v1_failure_predecessor": dict(v1_failure_predecessor)},
        )
        raise PhysicalContractError("C-Pre V2 synthetic lifecycle failed after immutable attempt") from exc


def execute_cpre_v2_metadata_only(*, root: Path, repo_root: Path, capability: object,
                                  paths_by_surface: Mapping[str, Sequence[Path]],
                                  resolved_metadata_facts: Mapping[str, object],
                                  anchor_window_authority: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
                                  session_name: Callable[[Path], str] | None = None) -> dict[str, str]:
    """Production-wired V2 metadata-only successor; it cannot import Torch or a model."""
    closure = source_closure(repo_root)
    assert_bound_cpre_v2_repair(closure)
    binding = _consume(capability, expected_root=plan.CPRE_V2_ROOT_RELATIVE, closure=closure,
                       repo_root=repo_root, supplied_root=root)
    facts = inventory.validate_resolved_metadata_facts(resolved_metadata_facts)
    v1_failure = binding.get("v1_failure_predecessor")
    _require(isinstance(v1_failure, Mapping), "V2 C-Pre V1 predecessor missing")
    if binding.get("mode") == "live":
        _require(v1_failure == _future_v1_cpre_failure_predecessor(repo_root),
                 "V2 C-Pre V1 predecessor changed after issuance")
        current_anchor, current_anchor_sha = _future_static_anchor(repo_root)
        held_authority = binding.get("anchor_window_authority")
        _require(current_anchor_sha == binding.get("anchor_score_descriptor_sha256")
                 and isinstance(held_authority, Mapping)
                 and _canonical_json_sha(_future_anchor_window_authority(anchor_payload=current_anchor))
                 == binding.get("anchor_window_authority_sha256"), "V2 C-Pre anchor authority drift")
        _require(anchor_window_authority is None or _canonical_json_sha(anchor_window_authority)
                 == binding.get("anchor_window_authority_sha256"), "caller may not replace V2 anchor authority")
        anchor_window_authority = held_authority
        _require(binding.get("path_map_sha256") == _path_map_sha256(paths_by_surface)
                 and binding.get("predecessors") == _future_predecessors(repo_root),
                 "V2 C-Pre live path/predecessor drift")
    _require(anchor_window_authority is not None and set(paths_by_surface) == set(plan.SURFACE_ORDER)
             and set(anchor_window_authority) == set(plan.SURFACE_ORDER)
             and binding.get("metadata_facts") == facts
             and binding.get("anchor_window_authority_sha256") == _canonical_json_sha(anchor_window_authority),
             "V2 C-Pre input authority drift")
    _require(root.name == Path(plan.CPRE_V2_ROOT_RELATIVE).name, "V2 C-Pre root basename drift")
    _exact_absent(root.parent, root.name); root.mkdir(mode=0o700)
    reservation = _capture_reserved_root(root) if binding.get("mode") == "live" else None
    attempt = receipts.publish_pair(root, "attempt.json", build_cpre_v2_attempt(
        closure=closure, v1_failure_predecessor=v1_failure,
        metadata_authority={"provider": "direct_metadata_only_nwb_timestamp_evalmask_trialstart_v2",
                            "resolved_metadata_facts": facts,
                            "anchor_window_authority_sha256": _canonical_json_sha(anchor_window_authority),
                            "target_values_read": False, "model_or_checkpoint_opened": False,
                            "cuda_initialized": False},
    ))
    stage = "metadata_provider"
    try:
        records: list[inventory.ChronologyMetadata] = []; evidence: dict[str, Any] = {}
        for surface in plan.SURFACE_ORDER:
            current = inventory.metadata_only_nwb_records(
                paths=paths_by_surface[surface], surface=surface, resolved_metadata_facts=facts,
                anchor_window_authority=anchor_window_authority[surface], session_name=session_name,
                coordinate_law="v2")
            records.extend(current)
            evidence[surface] = inventory.crosscheck_anchor_window_authority_v2(
                records=current, surface=surface, authority_by_session=anchor_window_authority[surface])
        metadata = inventory.inventory_dataset_v2(records)
        metadata.update({"metadata_provider": "direct_metadata_only_nwb_timestamp_evalmask_trialstart_v2",
                         "resolved_metadata_facts": facts, "anchor_window_crosschecks": evidence,
                         "model_or_checkpoint_opened": False, "cuda_initialized": False})
        descriptor = receipts.publish_pair(root, "metadata_inventory.json", metadata)
        stage = "terminal"
        if reservation is not None: _verify_reserved_root(root, reservation)
        terminal = receipts.publish_pair(root, "terminal.json", {
            "schema": "m2_cpre_terminal_v2", "status": "TERMINAL", "attempt_sha256": attempt.sha256,
            "metadata_inventory_sha256": descriptor.sha256, "v1_failure_predecessor": dict(v1_failure),
            "target_access": False, "target_values_read": False, "model_or_checkpoint_opened": False,
            "cuda_initialized": False, "source_closure": dict(closure),
        })
        receipts.verify_topology(root, bodies=("attempt.json", "metadata_inventory.json", "terminal.json"))
        return {"attempt": attempt.sha256, "inventory": descriptor.sha256, "terminal": terminal.sha256}
    except Exception as exc:
        _publish_cpre_failure_after_attempt(
            root=root, attempt_sha256=attempt.sha256, stage=stage,
            error=f"{type(exc).__name__}: {exc}", failure_schema="m2_cpre_failure_v2",
            progress={"target_access": False, "target_values_read": False,
                      "model_or_checkpoint_opened": False, "cuda_initialized": False,
                      "metadata_inventory_published": "metadata_inventory.json" in os.listdir(root),
                      "v1_failure_predecessor": dict(v1_failure)},
        )
        raise PhysicalContractError("C-Pre V2 metadata-only lifecycle failed after immutable attempt") from exc


def _cubic_activity_reconstruction(*, raw_session: Mapping[str, Any], raw_starts: Any) -> Any:
    """Independent cubic B3S audit only; its output never seeds A0 memory."""
    import numpy as np
    from scipy.interpolate import interp1d
    neural = np.ascontiguousarray(np.asarray(raw_session["neural"], dtype=np.float32))
    starts = np.ascontiguousarray(np.asarray(raw_starts, dtype=np.int64))
    _require(neural.ndim == 2 and starts.size > plan.CALIBRATION_TRIALS,
             "cubic reconstruction raw authority topology drift")
    rows: list[Any] = []
    for position, start in enumerate(starts):
        stop = int(starts[position + 1]) if position + 1 < starts.size else int(neural.shape[0])
        # This exactly follows the sealed calibration datamodule's activity
        # construction: full raw trial -> cubic [100,N].  eval_mask governs
        # scoring windows only and is forbidden from this B3S audit/operator.
        trial = np.ascontiguousarray(neural[int(start):stop], dtype=np.float32)
        _require(trial.shape[0] >= 4, "cubic B3S reconstruction needs four raw trial bins")
        source = np.linspace(0.0, 1.0, trial.shape[0])
        target = np.linspace(0.0, 1.0, plan.B3S_BINS)
        row = np.ascontiguousarray(
            interp1d(source, trial, axis=0, kind="cubic", fill_value="extrapolate")(target), dtype=np.float32)
        _require(row.shape == (plan.B3S_BINS, neural.shape[1]) and bool(np.isfinite(row).all()),
                 "cubic B3S reconstruction output drift")
        rows.append(row)
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def _production_a0_context_after_attempt(*, surface: str, torch: Any) -> dict[str, Any]:
    """Open the frozen loader only after A0 attempt/capability admission."""
    import numpy as np
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as anchor_core
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as anchor_physical
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as native_physical

    _require(surface in plan.SURFACE_ORDER, "unknown A0 surface")
    scheduler_attestation = attest_a0_runtime_scheduler(surface=surface)
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "A0 production allows only visible physical GPU0")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "A0 requires exactly one visible GPU (physical GPU0)")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    uuid_attestation = normalize_gpu0_uuid(getattr(properties, "uuid", ""))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    model, data_module, _task, metadata = load_frozen_model_and_data()
    _require(metadata.get("checkpoint_sha256") == plan.ANCHOR_CHECKPOINT_SHA256
             and metadata.get("normalization_sha256") == plan.ANCHOR_NORMALIZER_SHA256,
             "A0 frozen loader checkpoint/normalizer drift")
    model = model.to(device).eval()
    _require(int(getattr(model.student, "window_size", -1)) == plan.WINDOW_BINS,
             "A0 resolved frozen model must expose exact decoder W50")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if surface == "external_post30_local":
        dataset, raw_sessions = data_module.val_heldout_dataset, data_module.val_calib_heldout_sessions
    else:
        dataset, raw_sessions = data_module.train_dataset, data_module.train_calib_heldin_sessions
    _require(dataset is not None and raw_sessions is not None, "A0 surface dataset/raw authority absent")
    return {"torch": torch, "device": device, "model": model, "student": model.student,
            "dataset": dataset, "raw_sessions": raw_sessions, "metadata": metadata,
            "anchor_core": anchor_core, "anchor_physical": anchor_physical,
            "native_physical": native_physical, "scheduler_attestation": scheduler_attestation,
            "gpu_uuid_attestation": uuid_attestation}


def _run_a0_session_production(*, context: Mapping[str, Any], surface: str, session: str,
                               batch_size: int,
                               anchor_window_evidence: Mapping[str, Any]) -> tuple[list[A0Row], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """One production M2 session: static, true trial, chunk0, chunk50.

    This is intentionally a single causal stream per arm.  Raw bin processing
    is distinct from decoder calls: all bins after the raw first30 boundary
    advance the state transaction, while the decoder runs only at governed W50
    endpoint coordinates.
    """
    import numpy as np
    _require(int(batch_size) == plan.A0_FROZEN_DECODE_BATCH_SIZE,
             "A0 production session decode batch law drift")
    torch, device, dataset, student = (context[key] for key in ("torch", "device", "dataset", "student"))
    anchor_core, anchor_physical, native = (context[key] for key in (
        "anchor_core", "anchor_physical", "native_physical"))
    _require(getattr(student, "decoder_mode", None) == "coupled", "A0 coupled student drift")
    _require(int(getattr(student, "window_size", -1)) == plan.WINDOW_BINS,
             "A0 resolved frozen decoder window_size must be exactly W50")
    raw_neural, raw_starts, _theta, _rates, _linear_activities = native._native_trial_views(
        context["raw_sessions"][session], session=session)
    # The preceding line deliberately discards the legacy linear activities.
    sealed_activity = np.ascontiguousarray(
        np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32))
    _require(sealed_activity.shape[0] == raw_starts.size
             and sealed_activity.shape[1:] == (plan.B3S_BINS, raw_neural.shape[1]),
             "sealed cubic trialized activity/full raw chronology mismatch")
    cubic = _cubic_activity_reconstruction(raw_session=context["raw_sessions"][session], raw_starts=raw_starts)
    seed_direct = np.ascontiguousarray(sealed_activity[:plan.ACTIVITY_CAPACITY], dtype=np.float32)
    cubic_seed = np.ascontiguousarray(cubic[:plan.ACTIVITY_CAPACITY], dtype=np.float32)
    reconstruction_max = float(np.max(np.abs(cubic_seed.astype(np.float64) - seed_direct.astype(np.float64))))
    _require(reconstruction_max <= 1e-6, "cubic reconstruction audit exceeds sealed first30 tolerance")
    selected = anchor_physical._support_indices(dataset, session, plan.CARRIER_BUDGET)
    _require(np.array_equal(selected, np.arange(plan.CARRIER_BUDGET, dtype=np.int64)),
             "A0 M10 support must be exact chronological [0..9]")
    side_np, carrier = anchor_physical._ridge_side(dataset, session, selected)
    _require(carrier["selected_indices"] == list(plan.M10_CHRONOLOGICAL_SELECTED_INDICES)
             and carrier["selected_indices_sha256"] == plan.M10_CHRONOLOGICAL_SELECTED_INDICES_SHA256,
             "A0 anchor ridge selected-index evidence drift")
    side = torch.from_numpy(np.ascontiguousarray(side_np, dtype=np.float32)).unsqueeze(0).to(device)
    # The shared native helper provides raw chronology and governed post30 rows
    # when supplied the *sealed cubic* array, never its own linear activities.
    query_rows = native._query_trial_rows(
        dataset, session, raw_neural=raw_neural, raw_starts=raw_starts, activities=sealed_activity)
    query_starts = np.ascontiguousarray(np.concatenate([row["metric_starts"] for row in query_rows]), dtype=np.int64)
    _require(query_starts.size > 0 and bool(np.all(np.diff(query_starts) > 0)), "A0 governed post30 starts drift")
    query_targets = np.ascontiguousarray(np.stack(
        [dataset.covariate_data[session][int(start) + plan.WINDOW_BINS - 1] for start in query_starts], axis=0),
        dtype=np.float32)
    # W50, padded neural coordinate = raw endpoint coordinate.  The first
    # query begins later than raw boundary but must not shift the chunk origin.
    boundary = int(raw_starts[plan.CALIBRATION_TRIALS])
    _require(int(query_starts[0]) >= boundary + plan.WINDOW_BINS - 1, "W50 post30 first query boundary drift")
    query_index = {int(start): position for position, start in enumerate(query_starts)}
    full_anchor_starts = np.ascontiguousarray(
        [start for name, start in dataset.window_indices if name == session], dtype=np.int64)
    _require(full_anchor_starts.size > 0 and bool(np.all(np.diff(full_anchor_starts) > 0)),
             "A0 anchor full-query starts drift")
    seed_rows = [seed_direct[index] for index in range(plan.ACTIVITY_CAPACITY)]
    memories = {
        "true_trial": chunk_memory.SeededActivity30RollingMemory(first30_b3s=seed_rows, carrier_support_indices=selected),
        "chunk_phase0": chunk_memory.SeededActivity30RollingMemory(first30_b3s=seed_rows, carrier_support_indices=selected),
        "chunk_phase50": chunk_memory.SeededActivity30RollingMemory(first30_b3s=seed_rows, carrier_support_indices=selected),
    }
    streams = {
        "chunk_phase0": chunk_memory.FixedChunkStream(memory=memories["chunk_phase0"], phase=0,
                                                        calibration_boundary_bin=boundary),
        "chunk_phase50": chunk_memory.FixedChunkStream(memory=memories["chunk_phase50"], phase=50,
                                                         calibration_boundary_bin=boundary),
    }
    initial_stack = chunk_memory.activity_sha256(seed_rows)
    identity_cache: dict[str, tuple[Any, str]] = {}
    def identity(memory: Any) -> tuple[Any, str]:
        stack = memory.identity_sha256()
        if stack not in identity_cache:
            support = torch.from_numpy(np.ascontiguousarray(np.stack(memory.rows()), dtype=np.float32)).unsqueeze(0).to(device)
            with torch.inference_mode():
                value = student.compute_identity(support, side_features=side)
            value_np = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
            identity_cache[stack] = (value, anchor_core.array_sha256(value_np))
        return identity_cache[stack]
    static_identity, initial_b3s = identity(memories["true_trial"])
    # Record the governed endpoint and the exact identity existing *before*
    # its causal update.  We intentionally defer model forwards until after
    # the raw stream has been traversed: contiguous equal-identity requests
    # can then use the frozen batch law without changing state, row order, or
    # which identity each endpoint observed.
    requests: dict[str, list[IdentityDecodeRequest]] = {
        "static": [], "true_trial": [], "chunk_phase0": [], "chunk_phase50": [],
    }
    static_windows = np.asarray(dataset.neural_data[session], dtype=np.float32)
    max_coordinate = int(query_starts[-1])
    trial_ends = {int(raw_starts[position + 1] - 1 if position + 1 < raw_starts.size else raw_neural.shape[0] - 1): position
                  for position in range(plan.CALIBRATION_TRIALS, raw_starts.size)}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for coordinate in range(boundary, max_coordinate + 1):
        true_memory = memories["true_trial"]
        true_memory.begin_causal_transaction(arm="true_trial", coordinate=coordinate)
        for stream in streams.values():
            stream.begin_causal_bin(coordinate)
        if coordinate in query_index:
            ordinal = int(query_index[coordinate])
            arm_identity = {
                "static": (static_identity, initial_b3s),
                "true_trial": identity(true_memory),
                "chunk_phase0": identity(memories["chunk_phase0"]),
                "chunk_phase50": identity(memories["chunk_phase50"]),
            }
            for arm, (current_identity, identity_sha256) in arm_identity.items():
                requests[arm].append(IdentityDecodeRequest(
                    ordinal=ordinal, raw_window_start=int(coordinate), identity=current_identity,
                    identity_sha256=identity_sha256,
                ))
        if coordinate in trial_ends:
            position = trial_ends[coordinate]
            true_memory.append_after_causal_transaction(
                arm="true_trial", coordinate=coordinate, activity=sealed_activity[position])
        else:
            true_memory.close_prediction_without_update(arm="true_trial", coordinate=coordinate)
        raw_bin = raw_neural[coordinate]
        for stream in streams.values():
            stream.append_after_causal_bin(coordinate, raw_bin)
    finishes = {name: stream.finish() for name, stream in streams.items()}
    decoded: dict[str, Any] = {}
    decode_evidence: dict[str, dict[str, Any]] = {}
    for arm in ("static", "true_trial", "chunk_phase0", "chunk_phase50"):
        decoded[arm], decode_evidence[arm] = decode_identity_snapshot_segments(
            torch=torch, student=student, neural_source=static_windows, requests=requests[arm],
            device=device, batch_size=int(batch_size), behavior_scale=5.0,
            array_sha256=anchor_core.array_sha256,
        )
    elapsed = time.monotonic() - started
    first = {
        arm: anchor_core.array_sha256(np.ascontiguousarray(decoded[arm][:1], dtype=np.float32))
        for arm in ("static", "true_trial", "chunk_phase0", "chunk_phase50")
    }
    _require(len({first[key] for key in first}) == 1 and len(first) == 4,
             "static/true/chunk first prediction parity drift")
    target_sha = anchor_core.array_sha256(query_targets)
    start_sha = anchor_core.array_sha256(query_starts)
    peak_memory = int(max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)))
    rows: list[A0Row] = []
    for arm, phase, memory in (("static", None, memories["true_trial"]), ("true_trial", None, memories["true_trial"]),
                               ("chunk", 0, memories["chunk_phase0"]), ("chunk", 50, memories["chunk_phase50"])):
        key = "static" if arm == "static" else ("true_trial" if arm == "true_trial" else f"chunk_phase{phase}")
        pred = np.ascontiguousarray(decoded[key], dtype=np.float32)
        final_identity = initial_b3s if arm == "static" else identity(memory)[1]
        finish = finishes.get(key, {"partial_final_bins_discarded": 0})
        rows.append(A0Row(surface=surface, session_id=session, arm=arm, phase=phase,
                          r2=float(anchor_core.variance_weighted_r2(query_targets, pred)),
                          first_prediction_sha256=first[key], prediction_sha256=anchor_core.array_sha256(pred),
                          initial_b3s_identity_sha256=initial_b3s, final_b3s_identity_sha256=final_identity,
                          initial_activity_stack_sha256=initial_stack,
                          final_activity_stack_sha256=initial_stack if arm == "static" else memory.identity_sha256(),
                          query_starts_sha256=start_sha, target_window_sha256=target_sha,
                          window_count=int(query_starts.size), update_count=0 if arm == "static" else memory.update_count,
                          eviction_count=0 if arm == "static" else memory.update_count,
                          partial_final_bins_discarded=int(finish["partial_final_bins_discarded"]),
                          parameter_updates=0, target_gradients=0, wall_seconds=elapsed, peak_memory_bytes=peak_memory))
    carrier_evidence = {session: carrier}
    seed_evidence = {session: {"seed_source": "dataset.calib_trialized_neural_features[:30]", "activity_rows": 30,
                               "reconstruction_law": "frozen_cubic_interpolation",
                               "sealed_first30_activity_stack_sha256": initial_stack,
                               "used_seed_activity_stack_sha256": initial_stack,
                               "reconstructed_first30_activity_stack_sha256": chunk_memory.activity_sha256(cubic_seed),
                               "reconstructed_vs_sealed_max_abs": reconstruction_max}}
    required_window_fields = ("cpre_anchor_reference_int64_sha256", "anchor_score_ordered_window_starts_sha256",
                              "anchor_reference_window_count") if surface == "external_post30_local" else (
        "cpre_anchor_reference_int64_sha256", "anchor_score_ordered_window_starts_sha256",
        "anchor_reference_window_count", "within_post30_exact_anchor_parity")
    _require(all(field in anchor_window_evidence for field in required_window_fields),
             "bound anchor window evidence missing")
    cpre_reference_sha = str(anchor_window_evidence["cpre_anchor_reference_int64_sha256"])
    anchor_reference_sha = str(anchor_window_evidence["anchor_score_ordered_window_starts_sha256"])
    reference_count = int(anchor_window_evidence["anchor_reference_window_count"])
    cpre_start_sha = inventory.int_sequence_sha256(query_starts)
    if surface == "external_post30_local":
        _require(cpre_reference_sha == inventory.int_sequence_sha256(full_anchor_starts)
                 and reference_count == int(full_anchor_starts.size)
                 and bool(np.all(np.isin(query_starts, full_anchor_starts))),
                 "external governed post30 starts are not a bound deterministic full-query subset")
    else:
        _require(cpre_reference_sha == cpre_start_sha and reference_count == int(query_starts.size)
                 and bool(anchor_window_evidence["within_post30_exact_anchor_parity"]),
                 "within post30 anchor parity drift")
    query_evidence = {session: {"governed_post30_cpre_int64_sha256": cpre_start_sha,
                                "governed_post30_anchor_core_sha256": start_sha,
                                "governed_post30_window_count": int(query_starts.size),
                                "cpre_anchor_reference_int64_sha256": cpre_reference_sha,
                                "anchor_score_ordered_window_starts_sha256": anchor_reference_sha,
                                "anchor_reference_window_count": reference_count,
                                "governed_post30_subset_of_anchor_full_query": True,
                                "within_post30_exact_anchor_parity": (
                                    bool(anchor_window_evidence["within_post30_exact_anchor_parity"])
                                    if surface == "within_post30" else None),
                                "batch_decode_evidence": {
                                    "frozen_batch_size": int(batch_size),
                                    "arms": {arm: decode_evidence[arm] for arm in (
                                        "static", "true_trial", "chunk_phase0", "chunk_phase50")},
                                }}}
    return rows, carrier_evidence, seed_evidence, query_evidence


def execute_a0_shard_production(*, root: Path, repo_root: Path, capability: object, surface: str,
                                roster: Sequence[str], cpre_terminal_sha256: str,
                                static_anchor: Mapping[str, Any] | None = None,
                                anchor_window_evidence: Mapping[str, Mapping[str, Any]] | None = None,
                                batch_size: int = 32,
                                execution_profile: A0ExecutionProfile = V1_EXECUTION_PROFILE,
                                runtime_factory: Callable[[str], Mapping[str, Any]] | None = None,
                                session_replay_factory: Callable[..., tuple[list[A0Row], Mapping[str, Any],
                                                                            Mapping[str, Any], Mapping[str, Any]]] | None = None) -> dict[str, str]:
    """Full production A0 shard, callable only after opaque root admission.

    The immutable attempt is published before Torch/CUDA/full-loader access.
    It uses GPU0 only; GPU1 is neither enumerated nor selected.
    """
    _require(surface in plan.SURFACE_ORDER and root.name == Path(execution_profile.root_for(surface)).name,
             "A0 production root/surface drift")
    _require(int(batch_size) == plan.A0_FROZEN_DECODE_BATCH_SIZE,
             "A0 production must use the frozen decode batch law")
    closure = execution_profile.closure_builder(repo_root)
    execution_profile.closure_validator(closure)
    if execution_profile.capability_consumer is None:
        binding = _consume_a0(capability, surface=surface, closure=closure,
                              cpre_terminal_sha256=cpre_terminal_sha256, roster=roster,
                              repo_root=repo_root, supplied_root=root,
                              root_relative=execution_profile.root_for(surface))
    else:
        binding = execution_profile.capability_consumer(
            capability=capability, surface=surface, closure=closure,
            cpre_terminal_sha256=cpre_terminal_sha256, roster=roster,
            repo_root=repo_root, root=root)
    receipt_bindings = binding.get("receipt_bindings")
    _require(receipt_bindings is None or isinstance(receipt_bindings, Mapping),
             "A0 execution-profile receipt binding schema drift")
    if binding.get("mode") == "live":
        held_anchor = binding["static_anchor_payload"]
        held_windows = binding["query_window_authority"]
        cpre = _future_cpre_completion(repo_root=repo_root, expected_terminal_sha256=cpre_terminal_sha256,
                                       closure=closure if execution_profile.cpre_uses_current_closure else None)
        current_anchor, current_anchor_sha = _future_static_anchor(repo_root)
        _require(cpre["metadata_inventory_sha256"] == binding.get("cpre_metadata_inventory_sha256")
                 and current_anchor_sha == binding.get("static_anchor_descriptor_sha256")
                 and _canonical_json_sha(current_anchor) == binding.get("static_anchor_payload_sha256")
                 and _canonical_json_sha(_derive_a0_query_window_authority(
                     metadata=cpre["metadata"], anchor_payload=current_anchor,
                     surface=surface, roster=roster))
                 == binding.get("query_window_authority_sha256"),
                 "A0 held C-Pre/anchor/query authority drift")
        _require(static_anchor is None or _canonical_json_sha(static_anchor)
                 == binding["static_anchor_payload_sha256"],
                 "caller may not replace root-issued static anchor authority")
        _require(anchor_window_evidence is None or _canonical_json_sha(anchor_window_evidence)
                 == binding["query_window_authority_sha256"],
                 "caller may not replace root-issued query window authority")
        static_anchor = held_anchor
        anchor_window_evidence = held_windows
    _require(static_anchor is not None and anchor_window_evidence is not None,
             "A0 production needs root-issued static/window authority")
    _exact_absent(root.parent, root.name)
    root.mkdir(mode=0o700)
    reservation = _capture_reserved_root(root) if binding.get("mode") == "live" else None
    attempt = receipts.publish_pair(root, "attempt.json", build_a0_attempt(
        surface=surface, closure=closure, cpre_terminal_sha256=cpre_terminal_sha256, roster=roster,
        batch_size=plan.A0_FROZEN_DECODE_BATCH_SIZE, execution_profile=execution_profile))
    stage = "torch_import"
    try:
        if runtime_factory is None:
            import torch
            stage = "production_context"
            context = _production_a0_context_after_attempt(surface=surface, torch=torch)
        else:
            stage = "production_context"
            context = dict(runtime_factory(surface))
            torch = context.get("torch")
            _require(torch is not None and hasattr(torch, "cuda"), "typed runtime factory lacks torch/cuda facade")
        stage = "launch"
        launch = receipts.publish_pair(root, "launch.json", {
            "schema": execution_profile.launch_schema, "attempt_sha256": attempt.sha256,
            "device_profile": static_device_profile(), "scheduler_profile": scheduler_profile(surface),
            "frozen_decode_batch_size": plan.A0_FROZEN_DECODE_BATCH_SIZE,
            "resolved_window_size": plan.WINDOW_BINS,
            "runtime_scheduler_attestation": context["scheduler_attestation"],
            "gpu_uuid_attestation": context["gpu_uuid_attestation"],
            "cuda_initialized": bool(torch.cuda.is_initialized()), "parameter_updates": 0,
            "target_gradients": 0,
            **({"execution_profile_bindings": dict(receipt_bindings)} if receipt_bindings is not None else {}),
        })
        stage = "replay"
        all_rows: list[A0Row] = []
        carrier_evidence: dict[str, Any] = {}
        seed_evidence: dict[str, Any] = {}
        query_evidence: dict[str, Any] = {}
        for session in roster:
            if session_replay_factory is None:
                rows, carrier, seed, query = _run_a0_session_production(
                    context=context, surface=surface, session=session, batch_size=int(batch_size),
                    anchor_window_evidence=anchor_window_evidence[session])
            else:
                rows, carrier, seed, query = session_replay_factory(
                    context=context, surface=surface, session=session, batch_size=int(batch_size),
                    anchor_window_evidence=anchor_window_evidence[session])
            all_rows.extend(rows); carrier_evidence.update(carrier); seed_evidence.update(seed); query_evidence.update(query)
        stage = "input_authority"
        authority = build_a0_input_authority(surface=surface, roster=roster,
                                             cpre_terminal_sha256=cpre_terminal_sha256,
                                             static_anchor=static_anchor, carrier_evidence=carrier_evidence,
                                             activity_seed_evidence=seed_evidence, query_surface_evidence=query_evidence)
        input_descriptor = receipts.publish_pair(root, "input_authority.json", authority)
        prediction_counts = {session: next(row.window_count for row in all_rows if row.session_id == session)
                             for session in roster}
        stage = "replay_receipt"
        replay = build_a0_replay(surface=surface, roster=roster, rows=all_rows,
                                 input_authority_sha256=input_descriptor.sha256,
                                 actual_scored_prediction_count=prediction_counts)
        replay_descriptor = receipts.publish_pair(root, "replay.json", replay)
        stage = "terminal"
        if binding.get("mode") == "live":
            _require(attest_a0_runtime_scheduler(surface=surface) == binding.get("scheduler_attestation"),
                     "A0 terminal scheduler attestation drift")
            historical_cpre = _future_cpre_completion(
                repo_root=repo_root, expected_terminal_sha256=cpre_terminal_sha256,
                closure=closure if execution_profile.cpre_uses_current_closure else None)
            _require(historical_cpre["metadata_inventory_sha256"]
                     == binding.get("cpre_metadata_inventory_sha256"),
                     "A0 terminal historical C-Pre inventory drift")
            _verify_reserved_root(root, reservation if reservation is not None else {})
        if execution_profile.final_validator is not None:
            execution_profile.final_validator(binding=binding, closure=closure, repo_root=repo_root,
                                              root=root, surface=surface, stage="terminal")
        terminal = receipts.publish_pair(root, "terminal.json", build_a0_terminal(
            attempt_sha256=attempt.sha256, launch_sha256=launch.sha256,
            input_authority_sha256=input_descriptor.sha256, replay_sha256=replay_descriptor.sha256,
            replay=replay, closure=closure, execution_profile=execution_profile,
            receipt_bindings=receipt_bindings if isinstance(receipt_bindings, Mapping) else None))
        receipts.verify_terminal_xor(root, success_bodies=(
            "attempt.json", "launch.json", "input_authority.json", "replay.json", "terminal.json"))
        return {"attempt": attempt.sha256, "launch": launch.sha256, "input_authority": input_descriptor.sha256,
                "replay": replay_descriptor.sha256, "terminal": terminal.sha256}
    except Exception as exc:
        if binding.get("mode") == "live":
            # Re-read the immutable predecessor before publishing an A0
            # failure too; it remains historical evidence, distinct from the
            # current A0 source closure recorded in this failure prefix.
            _future_cpre_completion(repo_root=repo_root, expected_terminal_sha256=cpre_terminal_sha256,
                                    closure=closure if execution_profile.cpre_uses_current_closure else None)
        if execution_profile.final_validator is not None:
            execution_profile.final_validator(binding=binding, closure=closure, repo_root=repo_root,
                                              root=root, surface=surface, stage="failure")
        names = set(os.listdir(root))
        prefix = tuple(name for name in ("attempt.json", "launch.json", "input_authority.json", "replay.json")
                       if name in names)
        try:
            receipts.publish_failure_preserving_prefix(
                root=root, prefix_bodies=prefix, attempt_sha256=attempt.sha256,
                schema=execution_profile.failure_schema, stage=stage, error=f"{type(exc).__name__}: {exc}",
                progress={"target_optimization": False, "parameter_updates": 0, "target_gradients": 0,
                          "torch_import_attempted": stage != "torch_import", "cuda_runtime_attempted": stage != "torch_import",
                          "source_closure": dict(closure),
                          **({"execution_profile_bindings": dict(receipt_bindings)}
                             if isinstance(receipt_bindings, Mapping) else {})},
            )
        except Exception as failure_exc:
            raise PhysicalContractError("A0 production failed and failure receipt could not be legally published") from failure_exc
        raise PhysicalContractError("A0 production shard failed after immutable attempt") from exc
