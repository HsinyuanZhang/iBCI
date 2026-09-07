"""Fail-closed v2 paired launcher successor with complete failure evidence."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_subject_m_stagep_paired_live_launcher as l1
import track_b_v2_subject_m_stagep_paired_real_producer_v2 as producer


SCHEMA_DRY_PLAN = "track_b_v2_subject_m_stagep_paired_live_launcher_dry_v2"
SCHEMA_PREFLIGHT = "track_b_v2_subject_m_stagep_paired_live_launcher_preflight_v2"
SCHEMA_LAUNCH_CONTRACT = "track_b_v2_subject_m_stagep_child_launch_contract_v2"
SCHEMA_FAILURE_COMPLETION = "track_b_v2_subject_m_stagep_child_failure_completion_v2"
SCHEMA_FAILURE_TERMINAL = "track_b_v2_subject_m_stagep_parent_finalization_failure_terminal_v2"
SCHEMA_SNAPSHOT_CLEANUP = "track_b_v2_subject_m_stagep_private_snapshot_cleanup_v2"
STATUS_DRY = "NO_GO__V2_ADDENDUM_AND_LAUNCHER_PREFLIGHT_REQUIRED__NO_WRITE"
STATUS_PREFLIGHT = "ROOT_REVIEWED_V2_LAUNCHER_PREFLIGHT__DOES_NOT_ALONE_AUTHORIZE_EXECUTION"

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_paired_live_launcher_v2.py"
TEST = REPO_ROOT / "cebra_exploration/tests/test_track_b_v2_subject_m_stagep_paired_live_launcher_v2.py"
NOTE = REPO_ROOT / "cebra_exploration/docs/TRACK_B_V2_SUBJECT_M_STAGEP_PAIRED_LIVE_LAUNCHER_V2.md"
CONDA_PYTHON = l1.CONDA_PYTHON
RESULT_ROOT = producer.RESULT_ROOT
LAUNCHER_PREFLIGHT = RESULT_ROOT / "launcher_preflight_v2.json"
PAIRED_COMPLETION = RESULT_ROOT / "aggregate/paired_completion_v2.json"
COST_RECEIPT = l1.COST_RECEIPT
COST_RECEIPT_SHA256 = l1.COST_RECEIPT_SHA256
PAIR_ORDER = producer.PAIR_ORDER
ROOT_ENV = l1.ROOT_ENV
INTERNAL_TOKEN_ENV = "TRACK_B_STAGEP_V2_INTERNAL_CHILD_TOKEN"
CHILD_VIEW_ENV = "TRACK_B_STAGEP_V2_CHILD_VIEW"
FORBIDDEN_CHILD_ENV = l1.FORBIDDEN_CHILD_ENV
PRIVATE_SNAPSHOT_POLICY = dict(l1.PRIVATE_SNAPSHOT_POLICY)


class StagePPairedLiveLauncherV2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StagePPairedLiveLauncherV2Error(message)


def _canonical_bytes(value: Any) -> bytes:
    return base.canonical_json_bytes(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical_bytes(value))


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def implementation_closure() -> dict[str, Any]:
    files = {
        "v2_launcher_core": producer._source_binding(Path(__file__), label="v2 launcher core"),
        "v2_launcher_cli": producer._source_binding(CLI, label="v2 launcher CLI"),
        "v2_launcher_tests": producer._source_binding(TEST, label="v2 launcher tests"),
        "v2_launcher_note": producer._source_binding(NOTE, label="v2 launcher note"),
        "v2_producer_recursive_closure": producer.implementation_closure(),
        "settled_v1_launcher_core": producer._source_binding(
            Path(l1.__file__), label="settled v1 launcher core"),
    }
    return {"files": files, "closure_sha256": _sha_json(files)}


def _launch_contract_path(view: str) -> Path:
    require(view in PAIR_ORDER, "v2 launch-contract view invalid")
    return Path(producer._topology(view)["cell_root"]) / "launch_contract_v2.json"


def _forensic_stream_paths(view: str) -> tuple[Path, Path]:
    require(view in PAIR_ORDER, "v2 forensic stream view invalid")
    root = Path(producer._topology(view)["cell_root"]) / "forensic_stream_fallback"
    return root / "child_stdout.raw", root / "child_stderr.raw"


def _snapshot_cleanup_path(view: str) -> Path:
    require(view in PAIR_ORDER, "v2 snapshot-cleanup view invalid")
    return Path(producer._topology(view)["cell_root"]) / "private_snapshot_cleanup.json"


def _prospective_paths(view: str) -> tuple[Path, ...]:
    t = producer._topology(view)
    forensic_stdout, forensic_stderr = _forensic_stream_paths(view)
    paths = (
        _launch_contract_path(view), Path(t["start"]), Path(t["source"]),
        Path(t["target_access_attempt"]), Path(t["target_lineage"]), Path(t["target"]),
        _snapshot_cleanup_path(view),
        Path(t["encoder"]), Path(t["checkpoint"]), Path(t["embeddings"]),
        *(Path(t["scores"][f"{route}__{decoder}"])
          for route in producer.ROUTES for decoder in producer.DECODERS),
        Path(t["child_stdout"]), Path(t["child_stderr"]),
        forensic_stdout, forensic_stderr,
        Path(t["completion"]), Path(t["terminal"]),
    )
    result = tuple(Path(os.path.abspath(path)) for path in paths)
    require(len(result) == len(set(result)) == 22,
            "v2 view topology must have exact 22 distinct output pairs")
    return result


def canonical_topology() -> dict[str, Any]:
    all_paths = [path for view in PAIR_ORDER for path in _prospective_paths(view)] + [PAIRED_COMPLETION]
    require(len(all_paths) == len(set(map(lambda path: Path(os.path.abspath(path)), all_paths))) == 45,
            "v2 paired topology alias/collision")
    return {
        "per_view": {view: producer._topology(view) for view in PAIR_ORDER},
        "launcher_owned_by_view": {view: {
            "launch_contract": str(_launch_contract_path(view)),
            "snapshot_cleanup": str(_snapshot_cleanup_path(view)),
            "forensic_stdout": str(_forensic_stream_paths(view)[0]),
            "forensic_stderr": str(_forensic_stream_paths(view)[1]),
        } for view in PAIR_ORDER},
        "prospective_pair_count_per_view": 22, "total_prospective_pair_count": 45,
        "mandatory_preopen_target_attempt": True, "mandatory_prefit_target_lineage": True,
        "mandatory_stdout_and_stderr_raw_pairs_including_empty": True,
        "reserved_forensic_stdout_and_stderr_pairs_for_group_publication_failure": True,
        "mandatory_held_fd_snapshot_cleanup_receipt": True,
        "global_paired_completion": str(PAIRED_COMPLETION),
        "private_snapshot_policy": dict(PRIVATE_SNAPSHOT_POLICY),
    }


def _child_argv(view: str) -> list[str]:
    require(view in PAIR_ORDER, "v2 child view invalid")
    return [str(CONDA_PYTHON), str(CLI), "--execute", "--i-have-authorization",
            "--internal-child", "--view", view]


def _child_env_contract(view: str) -> dict[str, str]:
    return {"PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "1",
            ROOT_ENV: "1", CHILD_VIEW_ENV: view}


@contextmanager
def _legacy_launcher_context() -> Iterator[None]:
    names = {
        "producer": producer, "RESULT_ROOT": RESULT_ROOT, "LAUNCHER_PREFLIGHT": LAUNCHER_PREFLIGHT,
        "PAIRED_COMPLETION": PAIRED_COMPLETION, "CLI": CLI, "TEST": TEST, "NOTE": NOTE,
        "CONDA_PYTHON": CONDA_PYTHON, "SCHEMA_DRY_PLAN": SCHEMA_DRY_PLAN,
        "SCHEMA_PREFLIGHT": SCHEMA_PREFLIGHT, "SCHEMA_LAUNCH_CONTRACT": SCHEMA_LAUNCH_CONTRACT,
        "SCHEMA_FAILURE_COMPLETION": SCHEMA_FAILURE_COMPLETION,
        "SCHEMA_FAILURE_TERMINAL": SCHEMA_FAILURE_TERMINAL, "STATUS_DRY": STATUS_DRY,
        "STATUS_PREFLIGHT": STATUS_PREFLIGHT, "INTERNAL_TOKEN_ENV": INTERNAL_TOKEN_ENV,
        "CHILD_VIEW_ENV": CHILD_VIEW_ENV, "implementation_closure": implementation_closure,
        "_prospective_paths": _prospective_paths, "_launch_contract_path": _launch_contract_path,
        "canonical_topology": canonical_topology, "_child_argv": _child_argv,
        "_child_env_contract": _child_env_contract,
    }
    saved = {name: getattr(l1, name) for name in names}
    for name, value in names.items():
        setattr(l1, name, value)
    try:
        with producer._v2_runtime_context():
            yield
    finally:
        for name, value in saved.items():
            setattr(l1, name, value)


def build_dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA_DRY_PLAN, "status": STATUS_DRY, "pair_order": list(PAIR_ORDER),
        "canonical_result_root": str(RESULT_ROOT), "topology": canonical_topology(),
        "v1_failure_provenance": producer.validate_immutable_v1_failure(),
        "producer_addendum": {"path": str(producer.ADDENDUM_PATH),
                              "present": os.path.lexists(producer.ADDENDUM_PATH)},
        "launcher_preflight": {"path": str(LAUNCHER_PREFLIGHT),
                               "present": os.path.lexists(LAUNCHER_PREFLIGHT)},
        "isolated_child": {"python": str(CONDA_PYTHON), "cwd": str(REPO_ROOT),
                           "argv_by_view": {view: _child_argv(view) for view in PAIR_ORDER},
                           "environment_by_view": {view: _child_env_contract(view) for view in PAIR_ORDER}},
        "launcher_implementation_closure": implementation_closure(),
        "execute_enabled": False, "subprocess_started": False, "target_path_resolved": False,
        "target_opened": False, "formal_data_opened": False, "torch_imported": False,
        "cebra_imported": False, "gpu_queried": False, "write_performed": False,
    }


def _read_pair(path: Path, *, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._read_pair(path, label=label, expected_sha256=expected_sha256)


def _publish(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._publish(path, payload)


def _cost_identity() -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._cost_identity()


def _fresh_admissions() -> dict[str, Mapping[str, Any]]:
    return {view: l1.sealed_runtime.build_stagep_live_admission(view=view) for view in PAIR_ORDER}


def _target_authority_bindings(
        *, capabilities: Mapping[str, producer.ViewExecutionCapability]) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._target_authority_bindings(capabilities=capabilities)


def _launcher_preflight_payload(*, target_bindings: Mapping[str, Any]) -> dict[str, Any]:
    admissions = _fresh_admissions(); plan = producer.build_no_target_review_plan()
    capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    addendum = producer.load_execution_addendum(admissions=admissions)
    payload = {
        "schema": SCHEMA_PREFLIGHT, "status": STATUS_PREFLIGHT,
        "canonical_path": str(LAUNCHER_PREFLIGHT), "canonical_result_root": str(RESULT_ROOT),
        "pair_order": list(PAIR_ORDER), "canonical_topology": canonical_topology(),
        "producer_addendum_body_sha256": addendum["body_sha256"],
        "producer_closure_sha256": plan["implementation_closure"]["closure_sha256"],
        "launcher_implementation_closure": implementation_closure(),
        "admission_sha256_by_view": {view: capabilities[view].admission_sha256 for view in PAIR_ORDER},
        "official_preflight_body_sha256_by_view": {
            view: capabilities[view].official_preflight_body_sha256 for view in PAIR_ORDER},
        "cost_bound_cuda_identity": _cost_identity(),
        "target_authority_bindings_by_view": dict(target_bindings),
        "v9_preflight_sha256": producer.V9_PREFLIGHT_SHA256,
        "v1_failure_provenance": producer.validate_immutable_v1_failure(),
        "child_argv_by_view": {view: _child_argv(view) for view in PAIR_ORDER},
        "child_environment_by_view": {view: _child_env_contract(view) for view in PAIR_ORDER},
        "authorizes_target_or_GPU_by_itself": False, "target_opened": False,
        "formal_data_opened": False, "torch_imported": False, "cebra_imported": False,
    }
    return payload | {"launcher_preflight_payload_sha256": _sha_json(payload)}


def build_launcher_preflight_candidate() -> dict[str, Any]:
    admissions = _fresh_admissions(); plan = producer.build_no_target_review_plan()
    capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    return _launcher_preflight_payload(
        target_bindings=_target_authority_bindings(capabilities=capabilities))


def publish_launcher_preflight(*, i_have_independent_root_review: bool = False) -> dict[str, Any]:
    require(i_have_independent_root_review is True, "v2 launcher preflight requires root review")
    require(not os.path.lexists(LAUNCHER_PREFLIGHT) and
            not os.path.lexists(Path(f"{LAUNCHER_PREFLIGHT}.sha256")),
            "v2 launcher preflight pair must be fresh")
    LAUNCHER_PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    payload = build_launcher_preflight_candidate()
    binding = producer.publish_json_pair(LAUNCHER_PREFLIGHT, payload)
    try:
        require(build_launcher_preflight_candidate() == payload,
                "v2 launcher preflight launch/final drift")
        require(load_launcher_preflight()["payload"] == payload, "v2 launcher preflight reload drift")
        return binding
    except BaseException:
        producer._rollback_owned_raw_pair(binding)
        raise


def load_launcher_preflight(*, validate_target_bindings: bool = True) -> dict[str, Any]:
    loaded = _read_pair(LAUNCHER_PREFLIGHT, label="v2 launcher preflight")
    frozen = loaded["payload"].get("target_authority_bindings_by_view")
    require(isinstance(frozen, Mapping) and set(frozen) == set(PAIR_ORDER),
            "v2 launcher preflight target authority set drift")
    expected = build_launcher_preflight_candidate() if validate_target_bindings else \
        _launcher_preflight_payload(target_bindings=frozen)
    require(loaded["payload"] == expected, "v2 launcher preflight/current closure drift")
    return loaded


def require_all_outputs_fresh(*, only_view: str | None = None,
                              existing_launch_contract: Mapping[str, Any] | None = None) -> None:
    views = PAIR_ORDER if only_view is None else (only_view,)
    require(only_view is None or only_view in PAIR_ORDER, "v2 freshness view invalid")
    with _legacy_launcher_context():
        for view in views:
            for path in _prospective_paths(view):
                l1._validate_real_existing_chain(path.parent)
                if existing_launch_contract is not None and path == _launch_contract_path(view).resolve():
                    require(existing_launch_contract.get("path") == str(path) and
                            _valid_sha(existing_launch_contract.get("body_sha256")),
                            "v2 existing launch-contract binding drift")
                else:
                    require(not os.path.lexists(path) and not os.path.lexists(Path(f"{path}.sha256")),
                            f"v2 prospective output not fresh: {path}")
            snapshot = Path(producer._topology(view)["private_snapshot"])
            require(not os.path.lexists(snapshot), f"v2 private snapshot not fresh: {view}")
    require(not os.path.lexists(PAIRED_COMPLETION) and
            not os.path.lexists(Path(f"{PAIRED_COMPLETION}.sha256")),
            "v2 paired completion not fresh")


class ExecutionPrerequisites:
    def __init__(self, admissions: Mapping[str, Any], capabilities: Mapping[str, Any],
                 producer_addendum: Mapping[str, Any], launcher_preflight: Mapping[str, Any],
                 cost_identity: Mapping[str, Any], target_authority_bindings: Mapping[str, Any]):
        self.admissions = admissions; self.capabilities = capabilities
        self.producer_addendum = producer_addendum; self.launcher_preflight = launcher_preflight
        self.cost_identity = cost_identity; self.target_authority_bindings = target_authority_bindings


def validate_pretarget_prerequisites(
        *, only_view: str | None = None,
        existing_launch_contract: Mapping[str, Any] | None = None) -> ExecutionPrerequisites:
    admissions = _fresh_admissions(); plan = producer.build_no_target_review_plan()
    capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    addendum = producer.load_execution_addendum(admissions=admissions)
    preflight = load_launcher_preflight(validate_target_bindings=False)
    require(preflight["payload"]["producer_addendum_body_sha256"] == addendum["body_sha256"],
            "v2 preflight/addendum drift")
    cost = _cost_identity()
    require(cost == preflight["payload"]["cost_bound_cuda_identity"], "v2 cost identity drift")
    require_all_outputs_fresh(only_view=only_view, existing_launch_contract=existing_launch_contract)
    # Deliberately use only the immutable identities already frozen in the root-reviewed
    # preflight.  No target materializer, ledger, input-path, or target-authority helper is
    # callable in this phase.  Live target authority is resolved only after the durable
    # per-cell access-attempt pair exists.
    frozen = preflight["payload"].get("target_authority_bindings_by_view")
    require(isinstance(frozen, Mapping) and set(frozen) == set(PAIR_ORDER),
            "v2 frozen target authority/preflight set drift")
    return ExecutionPrerequisites(admissions, capabilities, addendum, preflight, cost, frozen)


def _resolve_live_target_authority_after_attempt(
        *, view: str, prerequisites: ExecutionPrerequisites,
        attempt_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve exactly one target authority, only after its immutable attempt exists."""
    require(view in PAIR_ORDER, "v2 post-attempt target view invalid")
    attempt_path = Path(producer._topology(view)["target_access_attempt"])
    durable = _read_pair(attempt_path, label=f"v2 {view} durable target access attempt",
                         expected_sha256=str(attempt_binding.get("body_sha256")))
    frozen = prerequisites.target_authority_bindings[view]
    payload = durable["payload"]
    bare = dict(payload); declared = bare.pop("target_access_attempt_payload_sha256", None)
    require(durable["body_sha256"] == attempt_binding.get("body_sha256") and
            durable["sidecar_sha256"] == attempt_binding.get("sidecar_sha256") and
            payload.get("schema") == producer.SCHEMA_TARGET_ACCESS_ATTEMPT and
            declared == _sha_json(bare) and payload.get("target_authority") == frozen and
            payload.get("target_opened_before_publication") is False and
            payload.get("next_operation_may_open_target") is True,
            "v2 durable target attempt/frozen authority drift")
    capability = prerequisites.capabilities[view]
    with _legacy_launcher_context():
        plan = l1.target_materializer.build_development_target_materializer_dry_plan(
            dataset="subject_m", view=view,
            outer_fold_id=str(capability.cell["outer_fold_id"]),
            target_session_id=producer.TARGET_SESSION_ID)
    require(plan.get("status") ==
            "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS" and
            plan.get("dataset") == "subject_m" and plan.get("view") == view and
            plan.get("target_session_id") == producer.TARGET_SESSION_ID and
            plan.get("target_data_opened") is False and plan == frozen,
            f"v2 {view} post-attempt live target authority/preflight drift")
    return dict(plan)


def _publish_attempt_then_resolve_live_target_authority(
        *, view: str, prerequisites: ExecutionPrerequisites,
        source: producer.SourceProducerOutput,
        source_binding: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    capability = prerequisites.capabilities[view]
    topology = producer._topology(view)
    payload = producer.build_target_access_attempt_payload(
        capability=capability, source_receipt_body_sha256=str(source_binding["body_sha256"]),
        source_payload_sha256=source.payload["source_payload_sha256"],
        launcher_preflight_body_sha256=prerequisites.launcher_preflight["body_sha256"],
        target_authority=prerequisites.target_authority_bindings[view], caller_pid=os.getpid())
    binding = _publish(Path(topology["target_access_attempt"]), payload)
    live = _resolve_live_target_authority_after_attempt(
        view=view, prerequisites=prerequisites, attempt_binding=binding)
    return binding, live


def _launch_contract_payload(*, view: str, token: str, prerequisites: ExecutionPrerequisites,
                             parent_pid: int | None = None) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._launch_contract_payload(
            view=view, token=token, prerequisites=prerequisites, parent_pid=parent_pid)


def _publish_launch_contract(*, view: str, token: str,
                             prerequisites: ExecutionPrerequisites) -> dict[str, Any]:
    return _publish(_launch_contract_path(view), _launch_contract_payload(
        view=view, token=token, prerequisites=prerequisites))


def _load_child_launch_contract(*, view: str, token: str) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._load_child_launch_contract(view=view, token=token)


def _verify_child_environment(*, view: str, token: str,
                              launch_contract: Mapping[str, Any]) -> None:
    with _legacy_launcher_context():
        l1._verify_child_environment(view=view, token=token, launch_contract=launch_contract)


def _verify_gpu_against_cost(cost: Mapping[str, Any]) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._verify_gpu_against_cost(cost)


def _start_payload(**kwargs: Any) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._start_payload(**kwargs)


def _cleanup_private_snapshot_after_success(
        *, view: str, target: producer.TargetProducerOutput) -> dict[str, Any]:
    """Unlink the verified snapshot while its original inode remains held open."""
    path = Path(os.path.abspath(producer._topology(view)["private_snapshot"]))
    contract = target.payload_inputs.get("private_snapshot")
    require(isinstance(contract, Mapping) and
            Path(os.path.abspath(str(contract.get("private_snapshot_path")))) == path and
            _valid_sha(contract.get("private_snapshot_sha256")) and
            _valid_sha(contract.get("private_snapshot_identity_sha256")) and
            type(contract.get("private_snapshot_byte_count")) is int,
            "v2 held-FD snapshot cleanup contract drift")
    parent_before = path.parent.lstat()
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                        getattr(os, "O_NOFOLLOW", 0))
    fd = -1
    try:
        parent_info = os.fstat(parent_fd)
        require((parent_info.st_dev, parent_info.st_ino) ==
                (parent_before.st_dev, parent_before.st_ino),
                "v2 snapshot cleanup parent identity drift")
        fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        before = os.fstat(fd)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mode,
                    before.st_mtime_ns, before.st_ctime_ns)
        require(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o444 and
                before.st_size == contract["private_snapshot_byte_count"] and
                _sha_json(identity) == contract["private_snapshot_identity_sha256"],
                "v2 snapshot cleanup held inode identity/mode drift")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        require(digest.hexdigest() == contract["private_snapshot_sha256"],
                "v2 snapshot cleanup held inode SHA drift")
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        require(stat.S_ISREG(named.st_mode) and
                (named.st_dev, named.st_ino) == (before.st_dev, before.st_ino),
                "v2 snapshot cleanup pathname/held inode drift")
        os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        after = os.fstat(fd)
        require((after.st_dev, after.st_ino) == (before.st_dev, before.st_ino) and
                after.st_nlink == 0 and not os.path.lexists(path),
                "v2 snapshot cleanup unlink did not remove canonical held inode")
        payload = {
            "schema": SCHEMA_SNAPSHOT_CLEANUP,
            "status": "PRIVATE_SNAPSHOT_CANONICAL_LINK_REMOVED_WHILE_VERIFIED_FD_HELD",
            "cell": dict(target.capability.cell),
            "private_snapshot_path": str(path),
            "private_snapshot_sha256": contract["private_snapshot_sha256"],
            "private_snapshot_byte_count": before.st_size,
            "private_snapshot_identity_sha256": contract["private_snapshot_identity_sha256"],
            "original_device": before.st_dev, "original_inode": before.st_ino,
            "original_mode": "0444", "link_count_after_unlink": after.st_nlink,
            "held_fd_identity_unchanged_after_unlink": True,
            "canonical_path_absent_after_unlink": True,
            "directory_fsynced_after_unlink": True,
            "formal_data_opened": False, "target_updates": 0,
        }
        payload["snapshot_cleanup_payload_sha256"] = _sha_json(payload)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)
    return _publish(_snapshot_cleanup_path(view), payload)


def _require_sua_terminal_before_pmua(*, launcher_preflight: Mapping[str, Any]) -> dict[str, Any]:
    with _legacy_launcher_context():
        terminal = l1._require_sua_terminal_before_pmua(launcher_preflight=launcher_preflight)
    cleanup = _load_output(_snapshot_cleanup_path("sua"), schema=SCHEMA_SNAPSHOT_CLEANUP,
                           label="SUA predecessor held-FD snapshot cleanup")
    require(terminal["payload"].get("snapshot_cleanup_receipt_body_sha256") ==
            cleanup["body_sha256"] and
            cleanup["payload"].get("link_count_after_unlink") == 0 and
            cleanup["payload"].get("canonical_path_absent_after_unlink") is True and
            not os.path.lexists(producer._topology("sua")["private_snapshot"]),
            "pMUA start forbidden before exact SUA held-FD snapshot cleanup receipt")
    return terminal


def _binding_subset(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("path", "body_sha256", "sidecar_sha256")}


def execute_internal_child(*, view: str, token: str) -> int:
    launch_contract = _load_child_launch_contract(view=view, token=token)
    _verify_child_environment(view=view, token=token, launch_contract=launch_contract)
    if view == "pseudo_mua":
        early = load_launcher_preflight(validate_target_bindings=False)
        _require_sua_terminal_before_pmua(launcher_preflight=early)
    prerequisites = validate_pretarget_prerequisites(
        only_view=view, existing_launch_contract=launch_contract)
    require(launch_contract["payload"] == _launch_contract_payload(
        view=view, token=token, prerequisites=prerequisites, parent_pid=os.getppid()),
        "v2 launch contract/current prerequisite drift")
    capability = prerequisites.capabilities[view]
    gpu_identity = _verify_gpu_against_cost(prerequisites.cost_identity)
    topology = producer._topology(view)
    start_payload = _start_payload(capability=capability, prerequisites=prerequisites,
                                   gpu_identity=gpu_identity, launch_contract=launch_contract)
    _publish(Path(topology["start"]), start_payload)
    source = producer.materialize_and_verify_strict27_source(capability)
    source_binding = _publish(Path(topology["source"]), source.payload)
    attempt_binding, _live_target_authority = _publish_attempt_then_resolve_live_target_authority(
        view=view, prerequisites=prerequisites, source=source, source_binding=source_binding)
    target = producer.materialize_target_from_private_snapshot(capability, source)
    lineage_payload = producer.build_prefit_target_lineage_payload(
        target=target, source_receipt_body_sha256=source_binding["body_sha256"],
        target_access_attempt_body_sha256=attempt_binding["body_sha256"])
    lineage_binding = _publish(Path(topology["target_lineage"]), lineage_payload)

    vendor = str(REPO_ROOT / "cebra_exploration/third_party/cebra")
    require(vendor not in sys.path, "vendored CEBRA unexpectedly preloaded before v2 fit")
    sys.path.insert(0, vendor)
    encoder = producer.fit_one_joint_encoder(source=source, target=target)
    persistence = producer.persist_sklearn_checkpoint_and_embeddings(
        capability=capability, estimator=encoder.estimator,
        embedding_arrays=encoder.persisted_embedding_arrays,
        reload_probe_inputs=encoder.reload_probe_inputs, cebra_loader=type(encoder.estimator).load)
    target_payload = dict(producer.finalize_target_payload(
        target=target, encoder=encoder, start_sha256=start_payload["start_payload_sha256"]))
    target_payload.pop("target_payload_sha256")
    target_payload.update({
        "target_access_attempt_receipt_body_sha256": attempt_binding["body_sha256"],
        "prefit_target_lineage_receipt_body_sha256": lineage_binding["body_sha256"],
        "prefit_target_lineage_payload_sha256": lineage_payload["target_lineage_payload_sha256"],
    })
    target_payload["target_payload_sha256"] = _sha_json(target_payload)
    _publish(Path(topology["target"]), target_payload)
    encoder_payload = producer.build_encoder_payload(
        capability=capability, start_sha256=start_payload["start_payload_sha256"],
        source_sha256=source.payload["source_payload_sha256"],
        target_sha256=target_payload["target_payload_sha256"],
        checkpoint_binding=persistence["checkpoint"], embedding_binding=persistence["embeddings"],
        fit_proof=encoder.fit_proof)
    encoder_payload = dict(encoder_payload) | {
        "source_receipt_path": topology["source"],
        "source_receipt_body_sha256": source_binding["body_sha256"],
        "prefit_target_lineage_receipt_body_sha256": lineage_binding["body_sha256"],
    }
    encoder_payload["encoder_payload_sha256"] = _sha_json(
        {key: value for key, value in encoder_payload.items() if key != "encoder_payload_sha256"})
    _publish(Path(topology["encoder"]), encoder_payload)
    scores = producer.score_all_six_readouts(
        encoder=encoder, target_payload=target_payload, encoder_payload=encoder_payload)
    for role, payload in scores.items():
        _publish(Path(topology["scores"][role]), payload)
    _cleanup_private_snapshot_after_success(view=view, target=target)
    return 0


def _load_output(path: Path, *, schema: str, label: str) -> dict[str, Any]:
    with _legacy_launcher_context():
        return l1._load_output(path, schema=schema, label=label)


def _load_raw_pair(path: Path, *, label: str) -> dict[str, Any]:
    body = producer._read_same_fd(path, label=f"{label} body", required_mode=0o444)
    side = producer._read_same_fd(Path(f"{path}.sha256"), label=f"{label} sidecar", required_mode=0o444)
    require(side.raw == f"{body.sha256}  {path.name}\n".encode("ascii"), f"{label} pair drift")
    return {"path": str(body.path), "body_sha256": body.sha256, "sidecar_sha256": side.sha256,
            "byte_count": len(body.raw), "mode": "0444"}


def _validate_v2_attempt_lineage(*, capability: Any, source: Mapping[str, Any],
                                 attempt: Mapping[str, Any], lineage: Mapping[str, Any],
                                 preflight: Mapping[str, Any]) -> None:
    ap = attempt["payload"]; lp = lineage["payload"]
    ap_bare = dict(ap); ap_declared = ap_bare.pop("target_access_attempt_payload_sha256", None)
    lp_bare = dict(lp); lp_declared = lp_bare.pop("target_lineage_payload_sha256", None)
    require(ap.get("schema") == producer.SCHEMA_TARGET_ACCESS_ATTEMPT and
            ap_declared == _sha_json(ap_bare) and
            ap.get("cell") == capability.cell and
            ap.get("source_receipt_body_sha256") == source["body_sha256"] and
            ap.get("source_payload_sha256") == source["payload"]["source_payload_sha256"] and
            ap.get("launcher_preflight_body_sha256") == preflight["body_sha256"] and
            ap.get("target_authority") == preflight["payload"][
                "target_authority_bindings_by_view"][capability.view] and
            ap.get("target_opened_before_publication") is False and
            ap.get("next_operation_may_open_target") is True,
            "v2 target-attempt chain drift")
    require(lp.get("schema") == producer.SCHEMA_TARGET_LINEAGE and
            lp_declared == _sha_json(lp_bare) and
            lp.get("cell") == capability.cell and
            lp.get("source_receipt_body_sha256") == source["body_sha256"] and
            lp.get("target_access_attempt_body_sha256") == attempt["body_sha256"] and
            lp.get("valid_starts_int64_sha256") == producer.V9_VALID_STARTS_SHA256 and
            lp.get("ordered_target_behavior_float32_sha256") == producer.V9_TARGET_SHA256 and
            lp.get("query_row_count") == producer.V9_QUERY_COUNT and
            lp.get("cebra_fit_started") is False and lp.get("gpu_fit_started") is False and
            lp.get("formal_data_opened") is False,
            "v2 prefit target-lineage chain drift")


def _finalize_success(*, view: str, returncode: int, argv: Sequence[str],
                      stdout_binding: Mapping[str, Any], stderr_binding: Mapping[str, Any]) -> dict[str, Any]:
    require(returncode == 0, "v2 cannot finalize nonzero child")
    preflight = load_launcher_preflight(); admissions = _fresh_admissions()
    plan = producer.build_no_target_review_plan()
    capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    addendum = producer.load_execution_addendum(admissions=admissions)
    live_closure = implementation_closure()
    require(preflight["payload"]["launcher_implementation_closure"] == live_closure and
            preflight["payload"]["producer_closure_sha256"] ==
            producer.implementation_closure()["closure_sha256"],
            "v2 launch/final/live closure drift")
    capability = capabilities[view]; t = producer._topology(view)
    launch = _load_output(_launch_contract_path(view), schema=SCHEMA_LAUNCH_CONTRACT,
                          label=f"{view} v2 launch contract")
    start = _load_output(Path(t["start"]), schema=producer.SCHEMA_START, label=f"{view} v2 start")
    source = _load_output(Path(t["source"]), schema=producer.SCHEMA_SOURCE, label=f"{view} v2 source")
    attempt = _load_output(Path(t["target_access_attempt"]), schema=producer.SCHEMA_TARGET_ACCESS_ATTEMPT,
                           label=f"{view} target attempt")
    lineage = _load_output(Path(t["target_lineage"]), schema=producer.SCHEMA_TARGET_LINEAGE,
                           label=f"{view} target lineage")
    cleanup = _load_output(_snapshot_cleanup_path(view), schema=SCHEMA_SNAPSHOT_CLEANUP,
                           label=f"{view} held-FD snapshot cleanup")
    target = _load_output(Path(t["target"]), schema=producer.SCHEMA_TARGET, label=f"{view} v2 target")
    encoder = _load_output(Path(t["encoder"]), schema=producer.SCHEMA_ENCODER, label=f"{view} v2 encoder")
    require(list(argv) == _child_argv(view), f"{view} v2 parent-observed argv drift")
    with _legacy_launcher_context():
        l1._validate_start_chain(view=view, start=start, capability=capability, preflight=preflight,
                                 addendum=addendum, launch_contract=launch)
    _validate_v2_attempt_lineage(capability=capability, source=source, attempt=attempt,
                                 lineage=lineage, preflight=preflight)
    cleanup_payload = cleanup["payload"]
    cleanup_bare = dict(cleanup_payload)
    cleanup_declared = cleanup_bare.pop("snapshot_cleanup_payload_sha256", None)
    snapshot_contract = lineage["payload"].get("private_snapshot", {})
    require(cleanup_declared == _sha_json(cleanup_bare) and
            cleanup_payload.get("cell") == capability.cell and
            cleanup_payload.get("private_snapshot_path") == t["private_snapshot"] and
            cleanup_payload.get("private_snapshot_sha256") ==
            snapshot_contract.get("private_snapshot_sha256") and
            cleanup_payload.get("private_snapshot_identity_sha256") ==
            snapshot_contract.get("private_snapshot_identity_sha256") and
            cleanup_payload.get("link_count_after_unlink") == 0 and
            cleanup_payload.get("held_fd_identity_unchanged_after_unlink") is True and
            cleanup_payload.get("canonical_path_absent_after_unlink") is True and
            not os.path.lexists(t["private_snapshot"]),
            "v2 held-FD snapshot cleanup receipt drift")
    require(target["payload"].get("target_access_attempt_receipt_body_sha256") == attempt["body_sha256"] and
            target["payload"].get("prefit_target_lineage_receipt_body_sha256") == lineage["body_sha256"] and
            encoder["payload"].get("prefit_target_lineage_receipt_body_sha256") == lineage["body_sha256"],
            "v2 target/encoder prefit lineage binding drift")
    checkpoint = _load_raw_pair(Path(t["checkpoint"]), label=f"{view} checkpoint")
    embeddings = _load_raw_pair(Path(t["embeddings"]), label=f"{view} embeddings")
    stdout_live = _load_raw_pair(Path(t["child_stdout"]), label=f"{view} child stdout")
    stderr_live = _load_raw_pair(Path(t["child_stderr"]), label=f"{view} child stderr")
    require(checkpoint["body_sha256"] == encoder["payload"]["checkpoint"]["body_sha256"] and
            embeddings["body_sha256"] == encoder["payload"]["embedding_bundle"]["body_sha256"] and
            all(stdout_live.get(key) == stdout_binding.get(key)
                for key in ("path", "body_sha256", "sidecar_sha256", "byte_count")) and
            all(stderr_live.get(key) == stderr_binding.get(key)
                for key in ("path", "body_sha256", "sidecar_sha256", "byte_count")),
            "v2 checkpoint/embedding/child-stream immutable binding drift")
    stdout_binding = stdout_live; stderr_binding = stderr_live
    scores: dict[str, Mapping[str, Any]] = {}; score_shas: dict[str, str] = {}; score_bindings = {}
    for route in producer.ROUTES:
        for decoder in producer.DECODERS:
            role = f"{route}__{decoder}"
            loaded = _load_output(Path(t["scores"][role]), schema=producer.SCHEMA_SCORE,
                                  label=f"{view} score {role}")
            scores[role] = loaded; score_shas[role] = loaded["payload"]["score_payload_sha256"]
            score_bindings[role] = _binding_subset(loaded)
    with _legacy_launcher_context():
        l1._validate_scientific_output_chain(
            view=view, capability=capability, start=start, source=source, target=target,
            encoder=encoder, scores=scores)
    completion = producer.build_completion_payload(
        capability=capability, start_sha256=start["payload"]["start_payload_sha256"],
        target_sha256=target["payload"]["target_payload_sha256"],
        encoder_sha256=encoder["payload"]["encoder_payload_sha256"],
        score_sha256_by_role=score_shas)
    completion = dict(completion); completion.pop("completion_payload_sha256")
    completion.update({
        "actual_exit_code": 0, "actual_started": True, "actual_argv": list(argv),
        "parent_launch_contract_body_sha256": launch["body_sha256"],
        "source_receipt_body_sha256": source["body_sha256"],
        "target_access_attempt_receipt_body_sha256": attempt["body_sha256"],
        "prefit_target_lineage_receipt_body_sha256": lineage["body_sha256"],
        "snapshot_cleanup_receipt_body_sha256": cleanup["body_sha256"],
        "child_stdout_raw_pair": dict(stdout_binding), "child_stderr_raw_pair": dict(stderr_binding),
        "launcher_preflight_body_sha256": preflight["body_sha256"],
        "launcher_closure_sha256_at_launch": start["payload"]["launcher_closure_sha256"],
        "launcher_closure_sha256_at_final": live_closure["closure_sha256"],
        "launcher_launch_final_live_exact_equal": True,
        "immutable_output_bindings": {
            "start": _binding_subset(start), "source": _binding_subset(source),
            "target_access_attempt": _binding_subset(attempt), "target_lineage": _binding_subset(lineage),
            "snapshot_cleanup": _binding_subset(cleanup),
            "target": _binding_subset(target), "encoder": _binding_subset(encoder),
            "checkpoint": checkpoint, "embeddings": embeddings, "scores": score_bindings,
            "stdout": dict(stdout_binding), "stderr": dict(stderr_binding)},
        "formal_data_opened": False, "target_updates": 0,
    })
    completion["completion_payload_sha256"] = _sha_json(completion)
    completion_binding = _publish(Path(t["completion"]), completion)
    terminal = producer.build_terminal_payload(
        capability=capability, completion_payload=completion,
        live_closure=producer.implementation_closure())
    terminal = dict(terminal) | {
        "launcher_preflight_body_sha256": preflight["body_sha256"],
        "launcher_closure_sha256_at_launch": start["payload"]["launcher_closure_sha256"],
        "launcher_closure_sha256_at_final": live_closure["closure_sha256"],
        "launcher_launch_final_live_exact_equal": True,
        "producer_launch_final_live_exact_equal": True,
        "parent_launch_contract_body_sha256": launch["body_sha256"],
        "source_receipt_body_sha256": source["body_sha256"],
        "target_access_attempt_receipt_body_sha256": attempt["body_sha256"],
        "prefit_target_lineage_receipt_body_sha256": lineage["body_sha256"],
        "snapshot_cleanup_receipt_body_sha256": cleanup["body_sha256"],
        "child_stdout_raw_pair": dict(stdout_binding), "child_stderr_raw_pair": dict(stderr_binding),
        "completion_receipt_body_sha256": completion_binding["body_sha256"], "actual_exit_code": 0,
    }
    terminal["terminal_payload_sha256"] = _sha_json(
        {key: value for key, value in terminal.items() if key != "terminal_payload_sha256"})
    _publish(Path(t["terminal"]), terminal)
    return terminal


ChildRunner = Callable[[str, Sequence[str], Mapping[str, str]], tuple[int, bytes, bytes]]


def _subprocess_child_runner(view: str, argv: Sequence[str], env: Mapping[str, str]) -> tuple[int, bytes, bytes]:
    done = subprocess.run(list(argv), cwd=REPO_ROOT, env=dict(env), capture_output=True, check=False)
    return int(done.returncode), bytes(done.stdout), bytes(done.stderr)


def _publish_stream_group(*, stdout_path: Path, stdout: bytes,
                          stderr_path: Path, stderr: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically publish two raw pairs as one four-link filesystem transaction."""
    bodies = (Path(os.path.abspath(stdout_path)), Path(os.path.abspath(stderr_path)))
    require(bodies[0] != bodies[1] and bodies[0].parent == bodies[1].parent,
            "v2 stream group requires two distinct paths in one parent")
    parent = bodies[0].parent
    require(parent.exists() and parent.is_dir() and not parent.is_symlink(),
            "v2 stream group parent must be a precreated real directory")
    parent_before = parent.lstat()
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                        getattr(os, "O_NOFOLLOW", 0))
    opened = os.fstat(parent_fd)
    require((opened.st_dev, opened.st_ino) == (parent_before.st_dev, parent_before.st_ino),
            "v2 stream group parent changed while opening")
    values = (("stdout", bodies[0], bytes(stdout)), ("stderr", bodies[1], bytes(stderr)))
    records: dict[str, dict[str, Any]] = {}
    created_temp: list[tuple[str, tuple[int, int]]] = []
    linked_final: list[tuple[str, tuple[int, int]]] = []

    def write_all(fd: int, raw: bytes) -> None:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(fd, remaining)
            require(written > 0, "v2 stream group short write")
            remaining = remaining[written:]

    def unlink_owned(name: str, identity: tuple[int, int]) -> None:
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == identity:
            os.unlink(name, dir_fd=parent_fd)

    try:
        # All four immutable inodes are complete before any canonical name is linked.
        nonce = secrets.token_hex(16)
        for role, body, raw in values:
            digest = _sha_bytes(raw)
            side_raw = f"{digest}  {body.name}\n".encode("ascii")
            item: dict[str, Any] = {"path": str(body), "body_sha256": digest,
                                    "sidecar_path": str(Path(f"{body}.sha256")),
                                    "sidecar_sha256": _sha_bytes(side_raw), "mode": "0444",
                                    "byte_count": len(raw)}
            for kind, content in (("body", raw), ("sidecar", side_raw)):
                temp_name = f".v2-stream-{nonce}-{role}-{kind}.tmp"
                fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
                info = os.fstat(fd)
                identity = (info.st_dev, info.st_ino)
                created_temp.append((temp_name, identity))
                try:
                    write_all(fd, content)
                    os.fsync(fd); os.fchmod(fd, 0o444); os.fsync(fd)
                finally:
                    os.close(fd)
                item[f"_{kind}_temp"] = temp_name
                item[f"{kind}_device"] = identity[0]
                item[f"{kind}_inode"] = identity[1]
            item["parent_device"] = opened.st_dev
            item["parent_inode"] = opened.st_ino
            records[role] = item
        os.fsync(parent_fd)
        for role, body, _raw in values:
            item = records[role]
            for kind, final_name in (("body", body.name), ("sidecar", f"{body.name}.sha256")):
                os.link(item[f"_{kind}_temp"], final_name, src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd, follow_symlinks=False)
                identity = (item[f"{kind}_device"], item[f"{kind}_inode"])
                linked_final.append((final_name, identity))
        os.fsync(parent_fd)
        parent_after = parent.lstat()
        require((parent_after.st_dev, parent_after.st_ino) == (opened.st_dev, opened.st_ino),
                "v2 stream group parent identity changed during publication")
        for final_name, identity in linked_final:
            info = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
            require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and
                    (info.st_dev, info.st_ino) == identity,
                    "v2 stream group final link identity/mode drift")
        for temp_name, identity in created_temp:
            unlink_owned(temp_name, identity)
        os.fsync(parent_fd)
        clean: list[dict[str, Any]] = []
        for role in ("stdout", "stderr"):
            item = {key: value for key, value in records[role].items() if not key.startswith("_")}
            clean.append(item)
        return clean[0], clean[1]
    except BaseException:
        for name, identity in reversed(linked_final):
            unlink_owned(name, identity)
        for name, identity in reversed(created_temp):
            unlink_owned(name, identity)
        os.fsync(parent_fd)
        raise
    finally:
        os.close(parent_fd)


def _observed_progress(view: str, topology: Mapping[str, Any]) -> dict[str, bool]:
    progress = {role: os.path.lexists(path) and os.path.lexists(Path(f"{path}.sha256"))
                for role, path in {
                    "start": topology["start"], "source": topology["source"],
                    "target_access_attempt": topology["target_access_attempt"],
                    "target_lineage": topology["target_lineage"], "target": topology["target"],
                    "encoder": topology["encoder"]}.items()}
    progress["snapshot_cleanup"] = (os.path.lexists(_snapshot_cleanup_path(view)) and
                                    os.path.lexists(Path(f"{_snapshot_cleanup_path(view)}.sha256")))
    progress["private_snapshot_present"] = os.path.lexists(topology["private_snapshot"])
    return progress


def _failure_payload(*, view: str, returncode: int, argv: Sequence[str], env: Mapping[str, str],
                     launch_binding: Mapping[str, Any], stdout_binding: Mapping[str, Any],
                     stderr_binding: Mapping[str, Any], progress: Mapping[str, bool],
                     stage: str, validation_error: str = "",
                     stream_storage_role: str = "canonical_child_stream_group") -> dict[str, Any]:
    target_may = bool(progress.get("target_access_attempt") or progress.get("target") or
                      progress.get("private_snapshot_present"))
    payload = {
        "schema": SCHEMA_FAILURE_COMPLETION, "status": "TERMINAL_V2_CHILD_FAILURE__CELL_NONREUSABLE",
        "cell": l1.sealed_runtime.StagePCell.from_view(view).as_dict(),
        "actual_exit_code": int(returncode), "actual_started": True, "actual_argv": list(argv),
        "actual_environment": {key: env.get(key) for key in _child_env_contract(view)},
        "parent_launch_contract": _binding_subset(launch_binding),
        "child_stdout_raw_pair": dict(stdout_binding), "child_stderr_raw_pair": dict(stderr_binding),
        "child_stream_storage_role": stream_storage_role,
        "child_stream_group_atomic": True,
        "stdout_sha256": stdout_binding["body_sha256"], "stderr_sha256": stderr_binding["body_sha256"],
        "stdout_byte_count": stdout_binding["byte_count"], "stderr_byte_count": stderr_binding["byte_count"],
        "failure_stage": stage, "validation_error_sha256": _sha_bytes(validation_error.encode()),
        "observed_progress_at_parent_reap": dict(progress),
        "target_opened_or_may_have_opened": target_may,
        "target_access_attempt_receipt_present": bool(progress.get("target_access_attempt")),
        "pMUA_start_permitted": False, "formal_data_opened": False, "target_updates": 0,
    }
    return payload | {"failure_completion_payload_sha256": _sha_json(payload)}


def _failure_terminal_payload(*, view: str, failure_completion: Mapping[str, Any],
                              completion_binding: Mapping[str, Any],
                              launch_binding: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA_FAILURE_TERMINAL,
        "status": "TERMINAL_V2_CHILD_FAILURE__CELL_NONREUSABLE__PMUA_BLOCKED",
        "cell": dict(failure_completion["cell"]),
        "actual_exit_code": int(failure_completion["actual_exit_code"]),
        "failure_stage": failure_completion["failure_stage"],
        "failure_completion": _binding_subset(completion_binding),
        "parent_launch_contract": _binding_subset(launch_binding),
        "source_receipt_present": bool(failure_completion["observed_progress_at_parent_reap"].get("source")),
        "target_access_attempt_receipt_present": bool(
            failure_completion["target_access_attempt_receipt_present"]),
        "target_lineage_receipt_present": bool(
            failure_completion["observed_progress_at_parent_reap"].get("target_lineage")),
        "snapshot_cleanup_receipt_present": bool(
            failure_completion["observed_progress_at_parent_reap"].get("snapshot_cleanup")),
        "target_opened_or_may_have_opened": bool(
            failure_completion["target_opened_or_may_have_opened"]),
        "child_stdout_raw_pair": dict(failure_completion["child_stdout_raw_pair"]),
        "child_stderr_raw_pair": dict(failure_completion["child_stderr_raw_pair"]),
        "child_stream_storage_role": failure_completion["child_stream_storage_role"],
        "formal_data_opened": False, "target_updates": 0,
        "pMUA_start_permitted": False, "cell_reusable": False,
    }
    return payload | {"failure_terminal_payload_sha256": _sha_json(payload)}


def _publish_failure_completion_and_terminal(
        *, view: str, failure: Mapping[str, Any], launch_binding: Mapping[str, Any]) -> dict[str, Any]:
    topology = producer._topology(view)
    completion_path = Path(topology["completion"])
    if os.path.lexists(completion_path):
        loaded = _read_pair(completion_path, label=f"v2 {view} preterminal completion")
        completion_binding = {key: loaded[key] for key in
                              ("path", "body_sha256", "sidecar_sha256")}
        completion_payload = loaded["payload"]
    else:
        completion_binding = _publish(completion_path, failure)
        completion_payload = dict(failure)
    # A success-shaped completion without its terminal is itself terminal failure evidence.
    if completion_payload.get("schema") != SCHEMA_FAILURE_COMPLETION:
        completion_payload = dict(failure) | {
            "prior_completion_schema": completion_payload.get("schema"),
            "prior_completion_body_sha256": completion_binding["body_sha256"]}
    terminal = _failure_terminal_payload(
        view=view, failure_completion=completion_payload,
        completion_binding=completion_binding, launch_binding=launch_binding)
    _publish(Path(topology["terminal"]), terminal)
    return terminal


def _run_one_child(*, view: str, token: str, runner: ChildRunner,
                   prerequisites: ExecutionPrerequisites) -> dict[str, Any]:
    argv = _child_argv(view)
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES", INTERNAL_TOKEN_ENV,
                          CHILD_VIEW_ENV, ROOT_ENV, *FORBIDDEN_CHILD_ENV}}
    env.update(_child_env_contract(view)); env[INTERNAL_TOKEN_ENV] = token
    launch_binding = _publish_launch_contract(view=view, token=token, prerequisites=prerequisites)
    returncode, stdout, stderr = runner(view, argv, env)
    t = producer._topology(view)
    try:
        stdout_binding, stderr_binding = _publish_stream_group(
            stdout_path=Path(t["child_stdout"]), stdout=stdout,
            stderr_path=Path(t["child_stderr"]), stderr=stderr)
        stream_role = "canonical_child_stream_group"
    except BaseException as stream_exc:
        forensic_stdout, forensic_stderr = _forensic_stream_paths(view)
        try:
            forensic_stdout.parent.mkdir(parents=True, exist_ok=True)
            stdout_binding, stderr_binding = _publish_stream_group(
                stdout_path=forensic_stdout, stdout=stdout,
                stderr_path=forensic_stderr, stderr=stderr)
        except BaseException as forensic_exc:
            raise StagePPairedLiveLauncherV2Error(
                "v2 child stream canonical group and forensic fallback publication both failed"
            ) from forensic_exc
        progress = _observed_progress(view, t)
        failure = _failure_payload(
            view=view, returncode=returncode, argv=argv, env=env, launch_binding=launch_binding,
            stdout_binding=stdout_binding, stderr_binding=stderr_binding, progress=progress,
            stage="raw_stream_group_publication_failure",
            validation_error=f"{type(stream_exc).__name__}:{stream_exc}",
            stream_storage_role="forensic_fallback_after_canonical_group_failure")
        _publish_failure_completion_and_terminal(
            view=view, failure=failure, launch_binding=launch_binding)
        raise StagePPairedLiveLauncherV2Error(
            f"{view} v2 child streams preserved in forensic fallback; cell nonreusable") from stream_exc
    progress = _observed_progress(view, t)
    if returncode != 0:
        failure = _failure_payload(
            view=view, returncode=returncode, argv=argv, env=env, launch_binding=launch_binding,
            stdout_binding=stdout_binding, stderr_binding=stderr_binding, progress=progress,
            stage="isolated_child_exit", stream_storage_role=stream_role)
        _publish_failure_completion_and_terminal(
            view=view, failure=failure, launch_binding=launch_binding)
        raise StagePPairedLiveLauncherV2Error(
            f"{view} v2 child failed exit {returncode}; raw stderr is immutable and pMUA blocked")
    try:
        return _finalize_success(view=view, returncode=0, argv=argv,
                                 stdout_binding=stdout_binding, stderr_binding=stderr_binding)
    except BaseException as exc:
        progress = _observed_progress(view, t)
        failure = _failure_payload(
            view=view, returncode=0, argv=argv, env=env, launch_binding=launch_binding,
            stdout_binding=stdout_binding, stderr_binding=stderr_binding, progress=progress,
            stage="post_exit_immutable_validation",
            validation_error=f"{type(exc).__name__}:{exc}", stream_storage_role=stream_role)
        if not os.path.lexists(t["terminal"]):
            _publish_failure_completion_and_terminal(
                view=view, failure=failure, launch_binding=launch_binding)
        raise StagePPairedLiveLauncherV2Error(
            f"{view} v2 child exited zero but finalization failed; pMUA blocked") from exc


def _orchestrate_authorized_pair(*, runner: ChildRunner) -> dict[str, Any]:
    require(os.environ.get(ROOT_ENV) == "1", f"v2 execution requires {ROOT_ENV}=1")
    prerequisites = validate_pretarget_prerequisites()
    sua = _run_one_child(view="sua", token=secrets.token_hex(32), runner=runner,
                         prerequisites=prerequisites)
    immutable_sua = _load_output(Path(producer._topology("sua")["terminal"]),
                                 schema=producer.SCHEMA_TERMINAL, label="v2 SUA terminal")
    require(immutable_sua["payload"] == sua and
            sua.get("status") == "TERMINAL_SUCCESS__DEVELOPMENT_PILOT_CELL",
            "v2 pMUA blocked before exact SUA terminal")
    pmua_prereq = validate_pretarget_prerequisites(only_view="pseudo_mua")
    pmua = _run_one_child(view="pseudo_mua", token=secrets.token_hex(32), runner=runner,
                          prerequisites=pmua_prereq)
    terminal_bindings = {
        view: _load_output(Path(producer._topology(view)["terminal"]),
                           schema=producer.SCHEMA_TERMINAL, label=f"v2 {view} terminal final")
        for view in PAIR_ORDER}
    final_preflight = load_launcher_preflight(); final_closure = implementation_closure()
    require(terminal_bindings["sua"]["payload"] == sua and
            terminal_bindings["pseudo_mua"]["payload"] == pmua and
            final_preflight["payload"]["launcher_implementation_closure"] == final_closure and
            all(binding["payload"].get("launcher_preflight_body_sha256") ==
                final_preflight["body_sha256"] and
                binding["payload"].get("launcher_closure_sha256_at_launch") ==
                final_closure["closure_sha256"] and
                binding["payload"].get("launcher_closure_sha256_at_final") ==
                final_closure["closure_sha256"] and
                binding["payload"].get("launcher_launch_final_live_exact_equal") is True and
                binding["payload"].get("producer_launch_final_live_exact_equal") is True
                for binding in terminal_bindings.values()),
            "v2 paired terminal launch/final/live chain drift")
    paired = producer.build_paired_completion_payload(sua_terminal=sua, pmua_terminal=pmua)
    paired = dict(paired) | {
        "launcher_preflight_body_sha256": final_preflight["body_sha256"],
        "launcher_closure_sha256": final_closure["closure_sha256"],
        "launcher_launch_final_live_exact_equal": True,
        "terminal_receipt_body_sha256_by_view": {
            view: terminal_bindings[view]["body_sha256"] for view in PAIR_ORDER},
        "v1_failure_completion_body_sha256": producer.V1_FAILURE_COMPLETION_BODY_SHA256,
    }
    paired["paired_completion_payload_sha256"] = _sha_json(
        {key: value for key, value in paired.items() if key != "paired_completion_payload_sha256"})
    _publish(PAIRED_COMPLETION, paired)
    return paired


def execute_authorized_parent() -> dict[str, Any]:
    return _orchestrate_authorized_pair(runner=_subprocess_child_runner)
